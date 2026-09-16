import unittest

from pipeline.signals import anomaly_signal, central_decision, model_signal, validate_signal


def candle(value, volume=10):
    return {'open':str(value), 'high':str(value*1.002), 'low':str(value*.998),
            'close':str(value*1.001), 'volume':str(volume)}


class SignalTests(unittest.TestCase):
    def test_model_contract(self):
        rows = [{'candle':{'data':candle(100+i,10+i)}} for i in range(48)]
        topic, event = model_signal('5m', rows, 1_000_000)
        validate_signal(event)
        self.assertEqual(topic, 'signals.model.v1')
        self.assertAlmostEqual(sum(event['output']['probabilities'].values()), 1)
        self.assertIn(event['output']['predicted_class'], ('HOLD','LONG','SHORT'))

    def test_anomaly_contract(self):
        candles = [candle(100,10) for _ in range(30)] + [candle(110,1000)]
        topic, event = anomaly_signal(candles, 1_000_000)
        validate_signal(event)
        self.assertEqual(topic, 'signals.anomaly.v1')
        self.assertTrue(event['output']['is_anomaly'])

    def test_central_holds_without_both_models(self):
        _, event = central_decision({}, 1_000_000)
        self.assertFalse(event['output']['trade_allowed'])
        self.assertEqual(event['output']['action'], 'HOLD')

    def test_critical_anomaly_blocks_trade(self):
        def model(source, timeframe):
            return {'signal_id':source, 'source':source, 'event_time_ms':1_000_000,
                    'output':{'probabilities':{'HOLD':.1,'LONG':.8,'SHORT':.1},'risk_ratio':2}}
        latest = {'model:5m':model('model:5m','5m'), 'model:15m':model('model:15m','15m'),
                  'anomaly:1m':{'signal_id':'a','source':'anomaly:1m','event_time_ms':1_000_000,
                                'output':{'anomaly_score':.9}}}
        _, event = central_decision(latest, 1_000_000)
        self.assertEqual(event['output']['action'], 'LONG')
        self.assertFalse(event['output']['trade_allowed'])
        self.assertIn('critical_1m_anomaly', event['output']['reasons'])


if __name__ == '__main__':
    unittest.main()
