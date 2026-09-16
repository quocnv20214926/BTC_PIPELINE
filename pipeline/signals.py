"""Versioned signal contracts and lightweight replaceable model logic."""
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

from .core import ANOMALY_SIGNALS, DECISIONS, INSTRUMENT, MODEL_SIGNALS, canonical, now_ms

SIGNAL_SCHEMA = 1
MODEL_ROOT = Path(os.getenv('MODEL_ROOT', 'models'))


def load_spec(relative_path):
    path = MODEL_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f'required model artifact not found: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _softmax(values):
    top = max(values)
    exp = [math.exp(v-top) for v in values]
    total = sum(exp)
    return [v/total for v in exp]


def _id(kind, source, timeframe, event_time_ms, payload):
    value = [SIGNAL_SCHEMA, kind, source, timeframe, event_time_ms, payload]
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def model_signal(timeframe, rows, event_time_ms, spec=None):
    """Create a stable model output contract from a 48-candle window.

    This deliberately simple model is a pipeline placeholder. A CNN adapter can
    later produce the same payload without changing downstream consumers.
    """
    spec = spec or load_spec(Path('m5/model.json') if timeframe == '5m' else Path('m15/realtime_adapter.json'))
    lookback = int(spec['lookback'])
    if timeframe not in ('5m', '15m') or len(rows) < lookback:
        raise ValueError('model signal requires 48 candles for 5m or 15m')
    if spec['timeframe'] != timeframe:
        raise ValueError('model artifact timeframe mismatch')
    candles = [r['candle']['data'] for r in rows[-lookback:]]
    close = [float(c['close']) for c in candles]
    volume = [float(c['volume']) for c in candles]
    returns = [(b/a)-1 for a, b in zip(close, close[1:])]
    momentum_4 = close[-1]/close[-5]-1
    momentum_12 = close[-1]/close[-13]-1
    ema = close[0]
    alpha = 2/21
    for value in close[1:]:
        ema = alpha*value + (1-alpha)*ema
    trend = close[-1]/ema-1
    volatility = statistics.pstdev(returns[-20:]) or 1e-6
    volume_mean = statistics.mean(volume[-20:-1])
    volume_std = statistics.pstdev(volume[-20:-1]) or 1e-6
    volume_z = (volume[-1]-volume_mean)/volume_std
    weights = spec['weights']
    directional_score = (weights['momentum_4']*momentum_4 +
                         weights['momentum_12']*momentum_12 +
                         weights['ema20_distance']*trend) / volatility
    hold_bias = float(spec['hold_bias'])
    hold_score = hold_bias - min(abs(directional_score), hold_bias)
    probabilities = _softmax([hold_score, directional_score, -directional_score])
    names = ('HOLD', 'LONG', 'SHORT')
    predicted = max(range(3), key=probabilities.__getitem__)
    # Conservative placeholder R:R; it is informational until calibrated by
    # paper-trading outcomes.
    risk_ratio = max(float(spec['risk_ratio_min']), min(float(spec['risk_ratio_max']),
                     abs(momentum_4)/(float(spec['risk_volatility_multiplier'])*volatility)))
    output = {
        'probabilities': dict(zip(names, probabilities)),
        'predicted_class': names[predicted], 'confidence': probabilities[predicted],
        'risk_ratio': risk_ratio,
        'features': {'momentum_4': momentum_4, 'momentum_12': momentum_12,
                     'ema20_distance': trend, 'volatility_20': volatility,
                     'volume_zscore_20': volume_z},
    }
    source = f'model:{timeframe}'
    event = {'schema_version': SIGNAL_SCHEMA, 'signal_type': 'model',
             'instrument': INSTRUMENT, 'source': source, 'timeframe': timeframe,
             'event_time_ms': event_time_ms, 'produced_at_ms': now_ms(),
             'model_version': spec['model_version'], 'model_type': spec['model_type'],
             'lookback': lookback, 'horizon_candles': int(spec['horizon_candles']), 'output': output}
    event['signal_id'] = _id('model', source, timeframe, event_time_ms, output)
    return MODEL_SIGNALS, event


def anomaly_signal(candles, event_time_ms, spec=None):
    """Detect unusual 1m return, range, and volume using prior observations."""
    spec = spec or load_spec(Path('m1/anomaly_detector.json'))
    lookback = int(spec['lookback'])
    if len(candles) < lookback + 1:
        raise ValueError('anomaly detector requires at least 31 candles')
    current, history = candles[-1], candles[-lookback-1:-1]
    current_return = float(current['close'])/float(current['open'])-1
    history_returns = [float(c['close'])/float(c['open'])-1 for c in history]
    current_range = (float(current['high'])-float(current['low']))/float(current['open'])
    history_ranges = [(float(c['high'])-float(c['low']))/float(c['open']) for c in history]
    current_log_volume = math.log1p(float(current['volume']))
    history_log_volume = [math.log1p(float(c['volume'])) for c in history]
    def zscore(value, values):
        return (value-statistics.mean(values))/(statistics.pstdev(values) or 1e-6)
    z = {'return_z': zscore(current_return, history_returns),
         'range_z': zscore(current_range, history_ranges),
         'volume_z': zscore(current_log_volume, history_log_volume)}
    score = min(1.0, max(abs(z['return_z']), abs(z['range_z']), max(z['volume_z'], 0))/float(spec['score_scale_z']))
    warning, critical = float(spec['warning_threshold']), float(spec['critical_threshold'])
    output = {**z, 'anomaly_score': score, 'is_anomaly': score >= warning,
              'severity': 'critical' if score >= critical else ('warning' if score >= warning else 'normal')}
    source = 'anomaly:1m'
    event = {'schema_version': SIGNAL_SCHEMA, 'signal_type': 'anomaly',
             'instrument': INSTRUMENT, 'source': source, 'timeframe': '1m',
             'event_time_ms': event_time_ms, 'produced_at_ms': now_ms(),
             'detector_version': spec['detector_version'], 'model_type': spec['model_type'],
             'lookback': lookback, 'output': output}
    event['signal_id'] = _id('anomaly', source, '1m', event_time_ms, output)
    return ANOMALY_SIGNALS, event


def central_decision(latest, decision_time_ms=None):
    """Fuse current M5/M15 model signals and the 1m anomaly risk gate."""
    decision_time_ms = decision_time_ms or now_ms()
    required = ('model:5m', 'model:15m')
    max_age = {'model:5m': 8*60_000, 'model:15m': 20*60_000}
    missing = [source for source in required if source not in latest]
    stale = [source for source in required if source in latest and
             decision_time_ms-latest[source]['event_time_ms'] > max_age[source]]
    reasons = []
    if missing or stale:
        action, confidence, trade_allowed = 'HOLD', 0.0, False
        reasons.append('missing:' + ','.join(missing))
        if stale: reasons.append('stale:' + ','.join(stale))
        probabilities = {'HOLD': 1.0, 'LONG': 0.0, 'SHORT': 0.0}
        rr = 0.0
    else:
        weights = {'model:5m': .40, 'model:15m': .60}
        probabilities = {name: sum(weights[s]*latest[s]['output']['probabilities'][name] for s in required)
                         for name in ('HOLD', 'LONG', 'SHORT')}
        action = max(probabilities, key=probabilities.get)
        confidence = probabilities[action]
        rr = sum(weights[s]*latest[s]['output']['risk_ratio'] for s in required)
        anomaly = latest.get('anomaly:1m')
        risk_gate = anomaly and anomaly['output']['anomaly_score'] >= .85
        trade_allowed = action != 'HOLD' and confidence >= .50 and rr >= 1.0 and not risk_gate
        if confidence < .50: reasons.append('low_confidence')
        if rr < 1.0: reasons.append('low_risk_ratio')
        if risk_gate: reasons.append('critical_1m_anomaly')
        if action == 'HOLD': reasons.append('ensemble_hold')
    output = {'action': action, 'trade_allowed': trade_allowed, 'confidence': confidence,
              'probabilities': probabilities, 'risk_ratio': rr, 'reasons': reasons,
              'input_signal_ids': {source: event['signal_id'] for source, event in latest.items()}}
    event = {'schema_version': SIGNAL_SCHEMA, 'decision_type': 'trade_intent',
             'instrument': INSTRUMENT, 'source': 'central:decision-v1',
             'event_time_ms': decision_time_ms, 'produced_at_ms': now_ms(), 'output': output}
    event['decision_id'] = _id('decision', event['source'], 'multi', decision_time_ms, output)
    return DECISIONS, event


def validate_signal(event):
    if event.get('schema_version') != SIGNAL_SCHEMA or event.get('instrument') != INSTRUMENT:
        raise ValueError('unsupported signal schema or instrument')
    if event.get('signal_type') not in ('model', 'anomaly'):
        raise ValueError('invalid signal type')
    if not event.get('signal_id') or not isinstance(event.get('output'), dict):
        raise ValueError('invalid signal envelope')
