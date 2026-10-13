import hashlib
import json
import unittest

from pipeline.core import canonical
from pipeline.public_site import chain, curve, monthly, page, summary, trade_pct

D = 86_400_000
T0 = 1_780_000_000_000


class PublicSiteTests(unittest.TestCase):
    def trades(self):
        eq, out = 1.0, []
        for i, r in enumerate([1.2, -1.13, -1.13, 1.2, -0.5]):
            risk = 0.01 * eq
            eq += risk * r
            out.append({'exit_close_ms': T0 + i * 10 * D, 'equity_after': eq, 'pnl_pct': trade_pct(risk, r, eq),
                        'result': 'TP' if r > 1 else 'SL'})
        return out

    def test_trade_pct_is_percent_of_account_before_close(self):
        self.assertAlmostEqual(trade_pct(0.01, 1.2, 1.012), 1.2)
        self.assertAlmostEqual(trade_pct(0.0102, -1.0, 1.02 - 0.0102), -1.0)

    def test_curve_drawdown_and_streak(self):
        t = self.trades()
        c = curve(t, 1.0, T0 - D)
        self.assertEqual(c['points'][0]['ret'], 0.0)
        self.assertAlmostEqual(c['total_ret'], 100 * (t[-1]['equity_after'] - 1))
        peak = t[0]['equity_after']
        self.assertAlmostEqual(c['max_dd'], 100 * (1 - t[2]['equity_after'] / peak))
        self.assertEqual(c['max_losing_streak'], 2)
        s = summary(t, [], c)
        self.assertEqual((s['trades'], s['wins'], s['losses']), (5, 2, 3))

    def test_monthly_compounds_within_month(self):
        m = monthly(self.trades())
        g = 1.0
        for x in m:
            g *= 1 + x['ret'] / 100
        self.assertAlmostEqual(100 * (g - 1), 100 * (self.trades()[-1]['equity_after'] - 1), places=1)
        self.assertEqual(sum(x['trades'] for x in m), 5)

    def test_hash_chain_is_reproducible(self):
        sig = [{'hour': f'h{i}', 'side': 'LONG', 'signal': i == 1, 'entered': i == 1, 'p_touch': 0.5, 'p_up': 0.6,
                'b': 0.01, 'skip': None} for i in range(3)]
        out = chain(sig)
        h = ''
        for s in sig:
            core = {k: s[k] for k in ('hour', 'side', 'signal', 'entered', 'p_touch', 'p_up', 'b')}
            h = hashlib.sha256((h + canonical(core)).encode()).hexdigest()
        self.assertEqual(out[-1]['hash'], h)
        tampered = chain([dict(sig[0], p_up=0.7)] + sig[1:])
        self.assertNotEqual(tampered[-1]['hash'], h)

    def test_page_is_complete_html(self):
        html = page()
        self.assertTrue(html.startswith('<!doctype html>'))
        self.assertIn('<html lang="vi">', html)
        self.assertIn("fetch('data.json'", html)


if __name__ == '__main__':
    unittest.main()
