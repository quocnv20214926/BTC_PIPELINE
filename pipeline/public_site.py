"""Worker `public-site`: trang công khai theo dõi kết quả chiến lược H1 (lệnh paper), tính theo % tài khoản.

Mỗi PUBLIC_INTERVAL_S giây (mặc định 300) và ngay khi có lệnh mới / lệnh đóng:
  1. Đọc strategy_trades, strategy_state, signal_events (CHỈ ĐỌC) -> data.json đã lọc, không có thông tin vận hành
     (không vốn thật, không token, không trạng thái máy chủ ngoài "giờ tín hiệu gần nhất").
  2. Ghi vào PUBLIC_DIR (mặc định artifacts/public): index.html (giao diện, pipeline/web/index.html) + data.json.
  3. Nếu có PUBLIC_GITHUB_REPO + PUBLIC_GITHUB_TOKEN: đẩy hai file lên GitHub (Contents API). Trang tĩnh phục vụ
     bằng GitHub Pages, Vercel hoặc Cloudflare Pages nối với repo đó. Lịch sử commit của GitHub là bằng chứng công khai,
     có dấu thời gian, rằng tín hiệu được công bố trước khi biết kết quả.

Cách tính % tài khoản (giống bộ thực thi): mỗi lệnh đặt rủi ro = risk_per_trade (1%) x vốn lúc vào lệnh, lãi kép.
  lãi / lỗ của lệnh (% tài khoản lúc đóng lệnh) = rủi ro x R sau phí / vốn trước khi đóng
  lợi nhuận luỹ kế = (vốn / vốn ban đầu - 1) x 100 ; sụt giảm = 1 - vốn / đỉnh vốn trước đó

Chuỗi hash: mỗi giờ tín hiệu h_i = sha256(h_{i-1} + JSON(giờ, hướng, có tín hiệu, vào lệnh, p_touch, p_lên, b)), h_0 = ''.
Ai cũng tính lại được từ data.json; kết hợp lịch sử commit -> không thể sửa / xoá tín hiệu cũ mà không lộ.

Biến môi trường:
  PUBLIC_DIR             artifacts/public
  PUBLIC_INTERVAL_S      300
  PUBLIC_SINCE           (YYYY-MM-DD) ngày bắt đầu forward test; mặc định = lệnh / tín hiệu đầu tiên
  PUBLIC_BACKTEST_CSV    artifacts/1h_strategy_v1/backtest_trades.csv  (phần backtest tham khảo; bỏ trống để tắt)
  PUBLIC_GITHUB_REPO     owner/repo   (bỏ trống = chỉ ghi file)
  PUBLIC_GITHUB_BRANCH   main
  PUBLIC_GITHUB_TOKEN    fine-grained token, quyền Contents: read/write trên repo đó
  PUBLIC_REPO_URL        link hiển thị trên trang (vd https://github.com/owner/repo/commits/main/data.json)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .core import INSTRUMENT, STRATEGY_SOURCE, canonical, now_ms
from .storage import connect, heartbeat

log = logging.getLogger(__name__)
H1 = 3_600_000
HERE = Path(__file__).resolve().parent / 'web'
CLOSED = ('TP', 'SL', 'ANOMALY', 'TIMEOUT')


def env(name, default, cast=str):
    v = os.getenv(name)
    return cast(default if v in (None, '') else v)


def iso(ms):
    return None if ms is None else datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat().replace('+00:00', 'Z')


def rnd(x, n=3):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), n)


# --------------------------------------------------------------------------- #
# Tính toán thuần (có unit test)
# --------------------------------------------------------------------------- #
def trade_pct(risk_frac: float, r_net: float, equity_after: float) -> float:
    """Lãi / lỗ của một lệnh, % tài khoản tại lúc đóng lệnh."""
    before = equity_after - risk_frac * r_net
    return 100.0 * risk_frac * r_net / before


def curve(trades: list[dict], start_equity: float = 1.0, start_ms: int | None = None) -> dict:
    """trades: lệnh đã đóng (exit_close_ms, equity_after) theo thứ tự đóng. Trả về đường vốn %, sụt giảm %, tóm tắt."""
    pts = [{'t': iso(start_ms), 'ret': 0.0, 'dd': 0.0}] if start_ms is not None else []
    peak, eq, max_dd = start_equity, start_equity, 0.0
    streak = worst = 0
    for t in trades:
        eq = float(t['equity_after'])
        peak = max(peak, eq)
        dd = 100.0 * (1 - eq / peak)
        max_dd = max(max_dd, dd)
        streak = streak + 1 if t['pnl_pct'] < 0 else 0
        worst = max(worst, streak)
        pts.append({'t': iso(t['exit_close_ms']), 'ret': rnd(100 * (eq / start_equity - 1), 3), 'dd': rnd(-dd, 3)})
    return {'points': pts, 'total_ret': 100 * (eq / start_equity - 1), 'max_dd': max_dd, 'max_losing_streak': worst}


def monthly(trades: list[dict], start_equity: float = 1.0) -> list[dict]:
    """Lợi nhuận theo tháng (lãi kép trong tháng), theo tháng ĐÓNG lệnh."""
    out, eq0, cur = [], start_equity, None
    for t in trades:
        m = datetime.fromtimestamp(t['exit_close_ms'] / 1000, tz=timezone.utc).strftime('%Y-%m')
        if cur is None or cur['month'] != m:
            if cur is not None:
                out.append(cur)
                eq0 = cur['_eq']
            cur = {'month': m, 'trades': 0, 'wins': 0, '_start': eq0, '_eq': eq0}
        cur['trades'] += 1
        cur['wins'] += t['pnl_pct'] > 0
        cur['_eq'] = float(t['equity_after'])
    if cur is not None:
        out.append(cur)
    return [{'month': c['month'], 'ret': rnd(100 * (c['_eq'] / c['_start'] - 1), 2), 'trades': c['trades'],
             'wins': c['wins']} for c in out]


def chain(signals: list[dict]) -> list[dict]:
    """Chuỗi hash của các giờ tín hiệu (đã sắp theo thời gian)."""
    h, out = '', []
    for s in signals:
        core = {k: s[k] for k in ('hour', 'side', 'signal', 'entered', 'p_touch', 'p_up', 'b')}
        h = hashlib.sha256((h + canonical(core)).encode()).hexdigest()
        out.append({**s, 'hash': h})
    return out


def summary(closed: list[dict], open_: list[dict], c: dict) -> dict:
    n = len(closed)
    wins = [t for t in closed if t['pnl_pct'] > 0]
    loss = [t for t in closed if t['pnl_pct'] <= 0]
    return {'trades': n, 'wins': len(wins), 'losses': len(loss),
            'win_rate': rnd(100 * len(wins) / n, 1) if n else None,
            'total_ret': rnd(c['total_ret'], 2), 'max_dd': rnd(c['max_dd'], 2),
            'avg_win': rnd(sum(t['pnl_pct'] for t in wins) / len(wins), 2) if wins else None,
            'avg_loss': rnd(sum(t['pnl_pct'] for t in loss) / len(loss), 2) if loss else None,
            'best': rnd(max((t['pnl_pct'] for t in closed), default=None), 2) if n else None,
            'worst': rnd(min((t['pnl_pct'] for t in closed), default=None), 2) if n else None,
            'max_losing_streak': c['max_losing_streak'], 'open': len(open_),
            'by_result': {k: sum(t['result'] == k for t in closed) for k in CLOSED}}


def backtest_section(csv_path: Path, risk: float) -> dict | None:
    """Backtest tham khảo (trades CSV của strategy_h1.export_bundle) -> % tài khoản, lãi kép rủi ro `risk`."""
    if not csv_path or not Path(csv_path).exists():
        return None
    import pandas as pd
    T = pd.read_csv(csv_path)
    T['exit'] = pd.to_datetime(T['exit_time'], utc=True)
    T['entry'] = pd.to_datetime(T['entry_time'], utc=True)
    T = T.sort_values('exit').reset_index(drop=True)
    # cùng cách tính vốn của backtest: rủi ro = risk x vốn lúc vào lệnh
    ev = sorted([(r.entry.value, 0, i) for i, r in T.iterrows()] + [(r.exit.value, 1, i) for i, r in T.iterrows()],
                key=lambda x: (x[0], -x[1]))
    eq, rk, after = 1.0, {}, {}
    for _, kind, i in ev:
        if kind == 0:
            rk[i] = risk * eq
        else:
            r = rk.pop(i, risk * eq)
            eq += r * float(T.at[i, 'R'])
            after[i] = (eq, r)
    rows = []
    for i, r in T.iterrows():
        e, rf = after[i]
        rows.append({'exit_close_ms': int(r.exit.value // 1_000_000), 'equity_after': e,
                     'pnl_pct': trade_pct(rf, float(r['R']), e), 'result': r['result']})
    rows.sort(key=lambda x: x['exit_close_ms'])
    c = curve(rows, 1.0, int(T['entry'].min().value // 1_000_000))
    years = {}
    for m in monthly(rows):
        y = m['month'][:4]
        years.setdefault(y, {'year': y, 'trades': 0, '_g': 1.0})
        years[y]['trades'] += m['trades']
        years[y]['_g'] *= 1 + m['ret'] / 100
    n = len(rows)
    step = max(1, len(c['points']) // 400)                  # rút gọn đường vốn cho trang
    pts = c['points'][::step] + ([c['points'][-1]] if (len(c['points']) - 1) % step else [])
    return {'from': iso(T['entry'].min().value // 1_000_000), 'to': iso(T['exit'].max().value // 1_000_000),
            'trades': n, 'win_rate': rnd(100 * sum(x['pnl_pct'] > 0 for x in rows) / n, 1),
            'total_ret': rnd(c['total_ret'], 1), 'max_dd': rnd(c['max_dd'], 1), 'points': pts,
            'years': [{'year': v['year'], 'ret': rnd(100 * (v['_g'] - 1), 1), 'trades': v['trades']}
                      for v in years.values()]}


# --------------------------------------------------------------------------- #
# Đọc DB
# --------------------------------------------------------------------------- #
def build(db, source: str = STRATEGY_SOURCE, since_ms: int | None = None, backtest: dict | None = None,
          repo_url: str | None = None, demo: bool = False) -> dict:
    st = db.execute('SELECT * FROM strategy_state WHERE source=%s', (source,)).fetchone()
    params = db.execute("""SELECT params FROM strategy_trades WHERE source=%s ORDER BY created_at DESC LIMIT 1""",
                        (source,)).fetchone()
    p = (params or {}).get('params') or {}
    risk = float(p.get('risk_per_trade', 0.01))
    first = db.execute('SELECT min(event_time_ms) AS t FROM signal_events WHERE source=%s', (source,)).fetchone()['t']
    since = since_ms if since_ms is not None else (first - H1 if first else None)
    rows = db.execute("""SELECT * FROM strategy_trades WHERE source=%s AND status <> 'CANCELLED'
        AND hour_open_ms >= %s ORDER BY hour_open_ms""", (source, since or 0)).fetchall()
    last_px = db.execute("""SELECT (data->>'close')::float AS c, period_start_ms FROM observations WHERE instrument=%s
        AND kind='candle' AND timeframe='1m' ORDER BY period_start_ms DESC LIMIT 1""", (INSTRUMENT,)).fetchone()
    closed, open_ = [], []
    for r in rows:
        side = 'LONG' if int(r['side']) > 0 else 'SHORT'
        slr = float(p.get('sl_mult', 0.75)) * float(r['b_pct'])
        base = {'signal_hour': iso(r['hour_open_ms']), 'side': side, 'entry_time': iso(r['entry_ms']),
                'entry': rnd(r['entry'], 1), 'tp': rnd(r['tp'], 1), 'sl': rnd(r['sl'], 1)}
        if r['status'] in CLOSED:
            pct = trade_pct(float(r['risk_frac']), float(r['r_net']), float(r['equity_after']))
            closed.append({**base, 'exit_time': iso(r['exit_close_ms']), 'exit': rnd(r['exit_price'], 1),
                           'result': r['status'], 'pnl_pct': rnd(pct, 3),
                           'hold_min': int((r['exit_close_ms'] - r['entry_ms']) // 60_000),
                           'exit_close_ms': int(r['exit_close_ms']), 'equity_after': float(r['equity_after'])})
        else:
            unreal = None
            if r['status'] == 'OPEN' and last_px and r['entry']:
                eq = float(st['equity']) if st else 1.0
                mark_r = int(r['side']) * math.log(last_px['c'] / float(r['entry'])) / slr
                unreal = rnd(100 * float(r['risk_frac']) * mark_r / eq, 2)
            open_.append({**base, 'status': 'Chờ khớp' if r['status'] == 'PENDING' else 'Đang mở',
                          'last_price': rnd(last_px['c'], 1) if last_px else None, 'unrealized_pct': unreal})
    closed.sort(key=lambda t: t['exit_close_ms'])
    c = curve(closed, 1.0, since)
    sig_rows = db.execute("""SELECT event_time_ms, payload FROM signal_events WHERE source=%s AND event_time_ms >= %s
        ORDER BY event_time_ms""", (source, since or 0)).fetchall()
    sigs = []
    for s in sig_rows:
        pl, d = s['payload'], s['payload']['decision']
        sg = pl.get('signal') or {}
        sigs.append({'hour': iso(pl['hour_open_ms']), 'side': d.get('side'), 'signal': bool(d.get('good')),
                     'entered': bool(d.get('admitted')), 'skip': d.get('skip_reason'),
                     'p_touch': rnd(sg.get('p_touch'), 4), 'p_up': rnd(sg.get('p_up'), 4),
                     'b': rnd(sg.get('b_pct'), 5)})
    sigs = chain(sigs)
    last = sig_rows[-1]['event_time_ms'] if sig_rows else None
    out = {
        'generated_at': iso(now_ms()), 'demo': demo, 'symbol': INSTRUMENT.split(':')[-1],
        'since': iso(since), 'last_signal_hour': iso(last - H1) if last else None,
        'running': bool(last and now_ms() - last < 2 * H1 + 15 * 60_000),
        'rules': {'risk_pct': risk * 100, 'tp_mult': p.get('tp_mult', 1.0), 'sl_mult': p.get('sl_mult', 0.75),
                  'anomaly_pct': 100 * float(p.get('anomaly_range', 0.0075)), 'max_positions': p.get('max_positions', 5),
                  'max_hold_days': int(p.get('max_hold_h', 720)) // 24, 'fee_pct': 100 * float(p.get('fee_pct', 0.001))},
        'summary': summary(closed, open_, c),
        'equity': c['points'], 'monthly': monthly(closed),
        'open': open_,
        'trades': [{k: v for k, v in t.items() if k not in ('exit_close_ms', 'equity_after')} for t in reversed(closed)],
        'signals': {'count': len(sigs), 'with_signal': sum(s['signal'] for s in sigs),
                    'entered': sum(s['entered'] for s in sigs), 'recent': sigs[-72:][::-1],
                    'chain_head': sigs[-1]['hash'] if sigs else None},
        'backtest': backtest, 'repo_url': repo_url,
    }
    return out


# --------------------------------------------------------------------------- #
# Xuất / đăng
# --------------------------------------------------------------------------- #
def write(out_dir: Path, data: dict) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'data.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    (out_dir / 'index.html').write_text(page(), encoding='utf-8')
    return [out_dir / 'index.html', out_dir / 'data.json']


def page() -> str:
    """Trang hoàn chỉnh để đặt trên hosting tĩnh (pipeline/web/index.html là phần thân)."""
    src = (HERE / 'index.html').read_text(encoding='utf-8')
    cut = src.index('<div class="wrap">')                # title, meta, font links, style -> <head>
    return ('<!doctype html>\n<html lang="vi">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>\n'
            + src[:cut] + '</head>\n<body>\n' + src[cut:] + '\n</body>\n</html>\n')


class GitHub:
    def __init__(self, repo: str, token: str, branch: str = 'main'):
        self.repo, self.branch = repo, branch
        self.c = httpx.Client(base_url='https://api.github.com', timeout=30,
                              headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json',
                                       'X-GitHub-Api-Version': '2022-11-28'})

    def put(self, path: str, content: bytes, message: str) -> bool:
        """Tạo / cập nhật file; bỏ qua nếu nội dung không đổi. Trả về True nếu có commit mới."""
        r = self.c.get(f'/repos/{self.repo}/contents/{path}', params={'ref': self.branch})
        sha = None
        if r.status_code == 200:
            cur = r.json()
            sha = cur['sha']
            if base64.b64decode(cur.get('content', '')) == content:
                return False
        elif r.status_code != 404:
            raise RuntimeError(f'GitHub GET {path}: HTTP {r.status_code}')
        body = {'message': message, 'content': base64.b64encode(content).decode(), 'branch': self.branch}
        if sha:
            body['sha'] = sha
        w = self.c.put(f'/repos/{self.repo}/contents/{path}', json=body)
        if w.status_code not in (200, 201):
            raise RuntimeError(f'GitHub PUT {path}: HTTP {w.status_code} {w.text[:200]}')
        return True


def public_worker(once: bool = False):
    out_dir = Path(env('PUBLIC_DIR', 'artifacts/public'))
    interval = env('PUBLIC_INTERVAL_S', '300', float)
    since = env('PUBLIC_SINCE', '')
    since_ms = int(datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp() * 1000) if since else None
    bt_csv = env('PUBLIC_BACKTEST_CSV', 'artifacts/1h_strategy_v1/backtest_trades.csv')
    repo, token = env('PUBLIC_GITHUB_REPO', ''), env('PUBLIC_GITHUB_TOKEN', '')
    gh = GitHub(repo, token, env('PUBLIC_GITHUB_BRANCH', 'main')) if repo and token else None
    repo_url = env('PUBLIC_REPO_URL', f'https://github.com/{repo}/commits' if repo else '') or None
    backtest = None
    try:
        backtest = backtest_section(Path(bt_csv), 0.01) if bt_csv else None
    except Exception as exc:  # noqa: BLE001 - phần backtest là tuỳ chọn
        log.warning('backtest section skipped: %s', exc)
    last_key = None
    with connect() as db:
        while True:
            try:
                key = db.execute("""SELECT count(*) AS n, max(updated_at) AS u FROM strategy_trades""").fetchone()
                sig = db.execute('SELECT max(event_time_ms) AS t FROM signal_events WHERE source=%s',
                                 (STRATEGY_SOURCE,)).fetchone()['t']
                k = (key['n'], str(key['u']), sig)
                data = build(db, since_ms=since_ms, backtest=backtest, repo_url=repo_url)
                files = write(out_dir, data)
                pushed = False
                if gh and k != last_key:
                    msg = f'cập nhật {data["generated_at"]}: {data["summary"]["trades"]} lệnh đã đóng, ' \
                          f'{data["summary"]["open"]} đang mở, chuỗi {(data["signals"]["chain_head"] or "")[:12]}'
                    for f in files:
                        pushed |= gh.put(f.name, f.read_bytes(), msg)
                last_key = k
                heartbeat(db, 'public-site', {'state': 'ok', 'trades': data['summary']['trades'],
                                              'pushed': pushed, 'github': bool(gh)})
            except Exception as exc:  # noqa: BLE001
                heartbeat(db, 'public-site', {'state': 'error', 'error': str(exc)[:300]})
                log.exception('public-site cycle failed')
                if once:
                    raise
            if once:
                return
            time.sleep(interval)
