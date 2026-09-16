import os
import unittest
from decimal import Decimal
from unittest.mock import patch

from pipeline.execution import _round_step, order_plan


class ExecutionRiskTests(unittest.TestCase):
    def test_long_plan_has_mandatory_sl_and_tp(self):
        with patch.dict(os.environ, {
            'TESTNET_ORDER_MARGIN_USDT':'100',
            'TESTNET_LEVERAGE':'50',
            'TESTNET_STOP_LOSS_PCT':'0.006',
            'TESTNET_TAKE_PROFIT_RR':'1.5'}):
            plan = order_plan('LONG', Decimal('100000'), Decimal('0.001'), Decimal('0.1'))
        self.assertEqual(plan['quantity'], Decimal('0.05'))
        self.assertEqual(plan['notional_usdt'], Decimal('5000'))
        self.assertEqual(plan['stop_loss'], Decimal('99400'))
        self.assertEqual(plan['take_profit'], Decimal('100900'))

    def test_short_plan_is_symmetric(self):
        with patch.dict(os.environ, {
            'TESTNET_ORDER_MARGIN_USDT':'100',
            'TESTNET_LEVERAGE':'50',
            'TESTNET_STOP_LOSS_PCT':'0.006',
            'TESTNET_TAKE_PROFIT_RR':'1.5'}):
            plan = order_plan('SHORT', Decimal('100000'), Decimal('0.001'), Decimal('0.1'))
        self.assertEqual(plan['stop_loss'], Decimal('100600'))
        self.assertEqual(plan['take_profit'], Decimal('99100'))

    def test_unsafe_risk_config_is_rejected(self):
        with patch.dict(os.environ, {'TESTNET_STOP_LOSS_PCT':'0.03'}):
            with self.assertRaises(ValueError):
                order_plan('LONG', Decimal('100000'), Decimal('0.001'), Decimal('0.1'))

    def test_empty_position_response_is_flat(self):
        from pipeline.execution import BinanceTestnet
        client = object.__new__(BinanceTestnet)
        client.request = lambda *args, **kwargs: []
        self.assertEqual(client.position()['side'], 'FLAT')


if __name__ == '__main__':
    unittest.main()
