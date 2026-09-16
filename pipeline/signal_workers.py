"""Kafka workers for model, anomaly, signal persistence, and central decisions."""
from collections import deque
import json
import logging
import time

from confluent_kafka import Consumer
from psycopg.types.json import Jsonb

from .core import ANOMALY_SIGNALS, DECISIONS, INSTRUMENT, MODEL_SIGNALS, RAW, READY
from .kafka_io import Sender, bootstrap
from .signals import anomaly_signal, central_decision, load_spec, model_signal, validate_signal
from .storage import connect, heartbeat

log = logging.getLogger(__name__)


def _consumer(group, topics, offset='earliest'):
    consumer = Consumer({'bootstrap.servers': bootstrap(), 'group.id': group,
                         'auto.offset.reset': offset, 'enable.auto.commit': False})
    consumer.subscribe(topics)
    return consumer


def model_worker(stop_after=None):
    """Turn completed M5/M15 feature sets into versioned model signals."""
    consumer, sender = _consumer('model-signal-worker-v1', [READY], offset='latest'), Sender()
    started = time.monotonic()
    try:
        with connect() as db:
            while stop_after is None or time.monotonic()-started < stop_after:
                message = consumer.poll(1)
                if message is None:
                    heartbeat(db, 'model-signals', {'state': 'idle'})
                    continue
                if message.error(): raise RuntimeError(str(message.error()))
                notice = json.loads(message.value())
                timeframe = notice.get('timeframe')
                if timeframe not in ('5m', '15m'):
                    consumer.commit(message=message, asynchronous=False)
                    continue
                row = db.execute('SELECT payload FROM feature_sets WHERE feature_set_id=%s',
                                 (notice['feature_set_id'],)).fetchone()
                if not row:
                    raise RuntimeError('feature set notice arrived before database row')
                rows = row['payload']['rows']
                if len(rows) < 48:
                    consumer.commit(message=message, asynchronous=False)
                    continue
                topic, event = model_signal(timeframe, rows, notice['window_end_ms'])
                sender.send(topic, INSTRUMENT, event); sender.flush()
                heartbeat(db, 'model-signals', {'state': 'ok', 'timeframe': timeframe,
                                                 'signal_id': event['signal_id']})
                consumer.commit(message=message, asynchronous=False)
    finally:
        consumer.close()


def anomaly_worker(stop_after=None):
    """Create rolling anomaly signals from every closed 1m candle."""
    consumer, sender = _consumer('anomaly-1m-worker-v1', [RAW], offset='latest'), Sender()
    spec = load_spec('m1/anomaly_detector.json')
    required = int(spec['lookback']) + 1
    candles = deque(maxlen=required)
    started = time.monotonic()
    try:
        with connect() as db:
            # Warm start từ DB để không phải đợi 31 phút và cũng không replay
            # toàn bộ raw history chỉ nhằm dựng rolling window.
            seed = db.execute('''SELECT data FROM observations
                WHERE instrument=%s AND timeframe='1m' AND kind='candle'
                ORDER BY period_start_ms DESC LIMIT %s''', (INSTRUMENT, required)).fetchall()
            for row in reversed(seed): candles.append(row['data'])
            while stop_after is None or time.monotonic()-started < stop_after:
                message = consumer.poll(1)
                if message is None:
                    heartbeat(db, 'anomaly-1m', {'state': 'warming' if len(candles)<31 else 'idle',
                                                 'candles': len(candles)})
                    continue
                if message.error(): raise RuntimeError(str(message.error()))
                event = json.loads(message.value())
                if event.get('kind') == 'candle' and event.get('timeframe') == '1m':
                    candles.append(event['data'])
                    if len(candles) == required:
                        topic, signal = anomaly_signal(list(candles), event['period_end_ms'], spec)
                        sender.send(topic, INSTRUMENT, signal); sender.flush()
                        heartbeat(db, 'anomaly-1m', {'state': 'ok', 'signal_id': signal['signal_id'],
                                                    'score': signal['output']['anomaly_score']})
                consumer.commit(message=message, asynchronous=False)
    finally:
        consumer.close()


def signal_writer(stop_after=None):
    """Persist the immutable signal/decision audit log and latest state."""
    topics = [MODEL_SIGNALS, ANOMALY_SIGNALS, DECISIONS]
    consumer = _consumer('signal-store-writer-v1', topics)
    started = time.monotonic()
    try:
        with connect() as db:
            while stop_after is None or time.monotonic()-started < stop_after:
                message = consumer.poll(1)
                if message is None:
                    heartbeat(db, 'signal-store', {'state': 'idle'})
                    continue
                if message.error(): raise RuntimeError(str(message.error()))
                event = json.loads(message.value())
                with db.transaction():
                    if message.topic() == DECISIONS:
                        db.execute('''INSERT INTO trade_decisions(decision_id,instrument,event_time_ms,
                            produced_at_ms,action,trade_allowed,payload) VALUES(%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT DO NOTHING''',
                            (event['decision_id'], event['instrument'], event['event_time_ms'],
                             event['produced_at_ms'], event['output']['action'],
                             event['output']['trade_allowed'], Jsonb(event)))
                    else:
                        validate_signal(event)
                        db.execute('''INSERT INTO signal_events(signal_id,instrument,signal_type,source,timeframe,
                            event_time_ms,produced_at_ms,payload) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT DO NOTHING''',
                            (event['signal_id'], event['instrument'], event['signal_type'], event['source'],
                             event['timeframe'], event['event_time_ms'], event['produced_at_ms'], Jsonb(event)))
                    heartbeat(db, 'signal-store', {'state': 'ok', 'topic': message.topic()})
                consumer.commit(message=message, asynchronous=False)
    finally:
        consumer.close()


def central_worker(stop_after=None):
    """Fuse the latest signals and publish a non-executing trade intent."""
    consumer, sender = _consumer('central-decision-worker-v1', [MODEL_SIGNALS, ANOMALY_SIGNALS]), Sender()
    latest = {}
    started = time.monotonic()
    with connect() as db:
        for row in db.execute('''SELECT DISTINCT ON(source) payload FROM signal_events
            ORDER BY source,event_time_ms DESC''').fetchall():
            event = row['payload']; latest[event['source']] = event
        try:
            while stop_after is None or time.monotonic()-started < stop_after:
                message = consumer.poll(1)
                if message is None:
                    heartbeat(db, 'central-decision', {'state': 'idle', 'sources': sorted(latest)})
                    continue
                if message.error(): raise RuntimeError(str(message.error()))
                event = json.loads(message.value()); validate_signal(event)
                current = latest.get(event['source'])
                if current is None or event['event_time_ms'] >= current['event_time_ms']:
                    latest[event['source']] = event
                    topic, decision = central_decision(latest, event['event_time_ms'])
                    sender.send(topic, INSTRUMENT, decision); sender.flush()
                    heartbeat(db, 'central-decision', {'state': 'ok',
                        'decision_id': decision['decision_id'], 'action': decision['output']['action'],
                        'trade_allowed': decision['output']['trade_allowed']})
                consumer.commit(message=message, asynchronous=False)
        finally:
            consumer.close()
