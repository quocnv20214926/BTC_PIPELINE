"""Fail-closed Binance USD-M Futures Testnet execution and position manager."""
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import hashlib
import hmac
import json
import logging
import os
import time
from urllib.parse import urlencode

import httpx
from confluent_kafka import Consumer
from psycopg.types.json import Jsonb

from .core import DECISIONS, EXECUTIONS, INSTRUMENT, canonical, now_ms
from .kafka_io import Sender, bootstrap
from .storage import connect, heartbeat

log = logging.getLogger(__name__)
# Signed query strings must never appear in normal container logs.
logging.getLogger('httpx').setLevel(logging.WARNING)
BASE_URL = 'https://testnet.binancefuture.com'
SYMBOL = 'BTCUSDT'


def _enabled():
    return os.getenv('TESTNET_EXECUTION_ENABLED', 'false').lower() in ('1', 'true', 'yes')


def _decimal_env(name, default):
    value = Decimal(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


class BinanceTestnet:
    def __init__(self):
        self.api_key = os.getenv('BINANCE_TESTNET_API_KEY', '').strip()
        self.secret = os.getenv('BINANCE_TESTNET_API_SECRET', '').strip().encode()
        if not self.api_key or not self.secret:
            raise RuntimeError('Binance Testnet credentials are missing')
        self.http = httpx.Client(base_url=BASE_URL, timeout=15,
                                 headers={'X-MBX-APIKEY': self.api_key})
        self.time_offset_ms = 0
        self.sync_time()

    def close(self):
        self.http.close()

    def sync_time(self):
        response = self.http.get('/fapi/v1/time')
        response.raise_for_status()
        self.time_offset_ms = int(response.json()['serverTime']) - now_ms()

    def request(self, method, path, params=None, signed=False):
        params = dict(params or {})
        if signed:
            params.update(timestamp=now_ms() + self.time_offset_ms, recvWindow=5000)
            query = urlencode(params)
            params['signature'] = hmac.new(self.secret, query.encode(), hashlib.sha256).hexdigest()
        response = self.http.request(method, path, params=params)
        if response.status_code >= 400:
            # Do not include request headers, query signature, or credentials in logs.
            raise RuntimeError(f'Binance {method} {path} failed ({response.status_code}): {response.text[:500]}')
        return response.json()

    def rules(self):
        info = self.request('GET', '/fapi/v1/exchangeInfo')
        symbol = next(item for item in info['symbols'] if item['symbol'] == SYMBOL)
        filters = {item['filterType']: item for item in symbol['filters']}
        return Decimal(filters['LOT_SIZE']['stepSize']), Decimal(filters['PRICE_FILTER']['tickSize'])

    def mark_price(self):
        return Decimal(self.request('GET', '/fapi/v1/premiumIndex', {'symbol': SYMBOL})['markPrice'])

    def position(self):
        try:
            rows = self.request('GET', '/fapi/v3/positionRisk', {'symbol': SYMBOL}, signed=True)
        except RuntimeError:
            rows = self.request('GET', '/fapi/v2/positionRisk', {'symbol': SYMBOL}, signed=True)
        if not rows:
            # A fresh Testnet account may return no row until BTCUSDT has its
            # first position. That is a valid flat account, not an error.
            return {'amount': Decimal('0'), 'side': 'FLAT',
                    'entry_price': Decimal('0'), 'unrealized_pnl': Decimal('0')}
        row = next((item for item in rows if item['symbol'] == SYMBOL and
                    item.get('positionSide', 'BOTH') == 'BOTH'), None)
        if row is None:
            raise RuntimeError('BTCUSDT One-way Mode is required; Hedge Mode is not supported')
        amount = Decimal(row['positionAmt'])
        return {'amount': amount, 'side': 'LONG' if amount > 0 else ('SHORT' if amount < 0 else 'FLAT'),
                'entry_price': Decimal(row['entryPrice']),
                'unrealized_pnl': Decimal(row.get('unRealizedProfit', row.get('unrealizedProfit', '0')))}

    def market_order(self, side, quantity, reduce_only=False):
        params = {'symbol': SYMBOL, 'side': side, 'type': 'MARKET',
                  'quantity': str(quantity), 'newOrderRespType': 'RESULT'}
        if reduce_only:
            params['reduceOnly'] = 'true'
        return self.request('POST', '/fapi/v1/order', params, signed=True)

    def configure_risk(self, leverage):
        if not 1 <= leverage <= 125:
            raise ValueError('TESTNET_LEVERAGE must be in range 1..125')
        try:
            self.request('POST', '/fapi/v1/marginType', {
                'symbol': SYMBOL, 'marginType': 'ISOLATED'}, signed=True)
        except RuntimeError as exc:
            # -4046 means the requested margin type is already active.
            if '-4046' not in str(exc):
                raise
        return self.request('POST', '/fapi/v1/leverage', {
            'symbol': SYMBOL, 'leverage': leverage}, signed=True)

    def protective_order(self, side, order_type, trigger_price):
        return self.request('POST', '/fapi/v1/algoOrder', {
            'algoType': 'CONDITIONAL', 'symbol': SYMBOL, 'side': side,
            'type': order_type, 'triggerPrice': str(trigger_price),
            'workingType': 'MARK_PRICE', 'closePosition': 'true',
            'priceProtect': 'true'
        }, signed=True)

    def open_algos(self):
        return self.request('GET', '/fapi/v1/openAlgoOrders', {'symbol': SYMBOL}, signed=True)

    def cancel_algos(self):
        return self.request('DELETE', '/fapi/v1/algoOpenOrders', {'symbol': SYMBOL}, signed=True)


def _round_step(value, step, rounding=ROUND_DOWN):
    return (value / step).to_integral_value(rounding=rounding) * step


def order_plan(side, mark_price, quantity_step, price_tick):
    margin = _decimal_env('TESTNET_ORDER_MARGIN_USDT', '100')
    leverage = int(os.getenv('TESTNET_LEVERAGE', '50'))
    if not 1 <= leverage <= 125:
        raise ValueError('TESTNET_LEVERAGE must be in range 1..125')
    notional = margin * leverage
    stop_pct = _decimal_env('TESTNET_STOP_LOSS_PCT', '0.006')
    rr = _decimal_env('TESTNET_TAKE_PROFIT_RR', '1.5')
    if stop_pct > Decimal('0.02') or rr < Decimal('1'):
        raise ValueError('unsafe testnet risk config: SL must be <=2% and R:R >=1')
    quantity = _round_step(notional / mark_price, quantity_step)
    if quantity <= 0:
        raise ValueError('notional is below the exchange minimum quantity')
    direction = Decimal('1') if side == 'LONG' else Decimal('-1')
    stop = _round_step(mark_price * (Decimal('1') - direction * stop_pct), price_tick, ROUND_HALF_UP)
    take = _round_step(mark_price * (Decimal('1') + direction * stop_pct * rr), price_tick, ROUND_HALF_UP)
    return {'quantity': quantity, 'stop_loss': stop, 'take_profit': take,
            'margin_usdt': margin, 'leverage': leverage, 'notional_usdt': notional,
            'stop_loss_pct': stop_pct, 'take_profit_rr': rr}


def _event(event_type, status, details):
    event_time = now_ms()
    payload = {'schema_version': 1, 'environment': 'binance-usdm-testnet',
               'instrument': INSTRUMENT, 'symbol': SYMBOL, 'event_type': event_type,
               'status': status, 'event_time_ms': event_time, 'details': details}
    payload['execution_id'] = hashlib.sha256(canonical(payload).encode()).hexdigest()
    return payload


def _record(db, sender, event_type, status, details):
    event = _event(event_type, status, details)
    db.execute('''INSERT INTO execution_events(execution_id,instrument,event_time_ms,event_type,status,payload)
                  VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',
               (event['execution_id'], INSTRUMENT, event['event_time_ms'], event_type, status, Jsonb(event)))
    sender.send(EXECUTIONS, INSTRUMENT, event)
    sender.flush()
    return event


def _save_state(db, position, protection='UNKNOWN', details=None, **ids):
    db.execute('''INSERT INTO execution_state(instrument,environment,position_side,position_amount,
        entry_price,unrealized_pnl,stop_loss_price,take_profit_price,entry_order_id,stop_order_id,
        take_profit_order_id,protection_status,last_sync_at,details)
        VALUES(%s,'binance-usdm-testnet',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),%s)
        ON CONFLICT(instrument) DO UPDATE SET position_side=excluded.position_side,
        position_amount=excluded.position_amount,entry_price=excluded.entry_price,
        unrealized_pnl=excluded.unrealized_pnl,stop_loss_price=excluded.stop_loss_price,
        take_profit_price=excluded.take_profit_price,entry_order_id=excluded.entry_order_id,
        stop_order_id=excluded.stop_order_id,take_profit_order_id=excluded.take_profit_order_id,
        protection_status=excluded.protection_status,last_sync_at=now(),details=excluded.details''',
        (INSTRUMENT, position['side'], position['amount'], position['entry_price'],
         position['unrealized_pnl'], ids.get('stop_loss'), ids.get('take_profit'),
         ids.get('entry_order_id'), ids.get('stop_order_id'), ids.get('take_profit_order_id'),
         protection, Jsonb(details or {})))


def _close_position(client, db, sender, position, reason):
    if position['side'] == 'FLAT':
        return
    client.cancel_algos()
    side = 'SELL' if position['side'] == 'LONG' else 'BUY'
    result = client.market_order(side, abs(position['amount']), reduce_only=True)
    _record(db, sender, 'POSITION_CLOSE', 'SUBMITTED', {
        'reason': reason, 'side': position['side'], 'quantity': str(abs(position['amount'])),
        'order_id': str(result.get('orderId', ''))})


def _open_position(client, db, sender, action, decision):
    quantity_step, price_tick = client.rules()
    mark = client.mark_price()
    plan = order_plan(action, mark, quantity_step, price_tick)
    leverage_result = client.configure_risk(plan['leverage'])
    entry_side = 'BUY' if action == 'LONG' else 'SELL'
    close_side = 'SELL' if action == 'LONG' else 'BUY'
    entry = client.market_order(entry_side, plan['quantity'])
    entry_id = str(entry.get('orderId', ''))
    try:
        stop = client.protective_order(close_side, 'STOP_MARKET', plan['stop_loss'])
        take = client.protective_order(close_side, 'TAKE_PROFIT_MARKET', plan['take_profit'])
    except Exception:
        # An entry without both exchange-hosted protections is never accepted.
        try:
            client.cancel_algos()
            live = client.position()
            _close_position(client, db, sender, live, 'PROTECTION_ORDER_FAILED')
        finally:
            _record(db, sender, 'ENTRY_ABORTED', 'ERROR', {'entry_order_id': entry_id,
                    'reason': 'protective_order_failed'})
        raise
    live = client.position()
    ids = {'entry_order_id': entry_id,
           'stop_order_id': str(stop.get('algoId', '')),
           'take_profit_order_id': str(take.get('algoId', '')),
           'stop_loss': plan['stop_loss'], 'take_profit': plan['take_profit']}
    _save_state(db, live, 'PROTECTED', {'decision_id': decision['decision_id']}, **ids)
    _record(db, sender, 'POSITION_OPEN', 'PROTECTED', {
        'decision_id': decision['decision_id'], 'side': action,
        'quantity': str(plan['quantity']), 'mark_price': str(mark),
        'margin_usdt': str(plan['margin_usdt']), 'notional_usdt': str(plan['notional_usdt']),
        'leverage': plan['leverage'], 'exchange_leverage': leverage_result.get('leverage'),
        'entry_price': str(live['entry_price']), 'stop_loss': str(plan['stop_loss']),
        'take_profit': str(plan['take_profit']), **{k: str(v) for k, v in ids.items() if 'price' not in k}})


def execution_worker():
    """Consume only fresh decisions and manage one testnet position at a time."""
    consumer = Consumer({'bootstrap.servers': bootstrap(), 'group.id': 'testnet-executor-v1',
                         'auto.offset.reset': 'latest', 'enable.auto.commit': False})
    consumer.subscribe([DECISIONS])
    sender = Sender()
    if not _enabled():
        with connect() as db:
            while True:
                heartbeat(db, 'testnet-executor', {'state': 'disabled',
                          'message': 'Set TESTNET_EXECUTION_ENABLED=true after adding testnet credentials'})
                time.sleep(10)
    client = BinanceTestnet()
    max_age_ms = int(os.getenv('TESTNET_MAX_DECISION_AGE_SECONDS', '90')) * 1000
    try:
        with connect() as db:
            last_sync = 0.0
            while True:
                message = consumer.poll(1)
                if time.monotonic() - last_sync >= 10:
                    position = client.position()
                    algos = client.open_algos() if position['side'] != 'FLAT' else []
                    protective = [a for a in algos if a.get('orderType') in ('STOP_MARKET', 'TAKE_PROFIT_MARKET')]
                    protection = 'PROTECTED' if len(protective) >= 2 else ('NONE' if position['side'] == 'FLAT' else 'MISSING')
                    _save_state(db, position, protection, {'open_algo_orders': len(algos)})
                    if position['side'] != 'FLAT' and protection == 'MISSING':
                        _record(db, sender, 'SAFETY_INTERLOCK', 'TRIGGERED', {'reason': 'position_without_sl_tp'})
                        _close_position(client, db, sender, position, 'POSITION_WITHOUT_SL_TP')
                    heartbeat(db, 'testnet-executor', {'state': 'ok', 'position': position['side'],
                                                       'protection': protection})
                    last_sync = time.monotonic()
                if message is None:
                    continue
                if message.error():
                    raise RuntimeError(str(message.error()))
                decision = json.loads(message.value())
                age = now_ms() - int(decision['produced_at_ms'])
                output = decision['output']
                if age < 0 or age > max_age_ms:
                    _record(db, sender, 'DECISION_REJECTED', 'STALE', {
                        'decision_id': decision['decision_id'], 'age_ms': age})
                elif output.get('trade_allowed'):
                    position = client.position()
                    action = output['action']
                    if position['side'] == 'FLAT':
                        _open_position(client, db, sender, action, decision)
                    elif position['side'] != action:
                        _close_position(client, db, sender, position, 'MODEL_REVERSAL')
                    else:
                        _record(db, sender, 'DECISION_IGNORED', 'POSITION_ALREADY_OPEN', {
                            'decision_id': decision['decision_id'], 'side': position['side']})
                consumer.commit(message=message, asynchronous=False)
    except Exception as exc:
        log.exception('testnet executor stopped safely')
        try:
            with connect() as db:
                heartbeat(db, 'testnet-executor', {'state': 'error', 'error': str(exc)[:500]})
        finally:
            raise
    finally:
        consumer.close()
        client.close()
