import asyncio
import json
import logging
import os
import random
import time

import httpx
from websockets.asyncio.client import connect as ws_connect

from .core import (COLLECT_KINDS_BY_TIMEFRAME, END_STAMPED_KINDS, INTERVALS, INSTRUMENT, RAW, candle, metric,
                   now_ms, validate)
from .kafka_io import Sender
from .storage import connect, heartbeat

# Logger dùng chung cho cả hai nguồn dữ liệu REST và WebSocket. Các log ở đây
# giúp phân biệt lỗi mạng, lỗi phản hồi từ Binance và trạng thái kết nối realtime.
log = logging.getLogger(__name__)
# BASE là máy chủ Binance Futures; WS có thể được ghi đè khi chạy trong môi
# trường kiểm thử hoặc khi cần kết nối tới một endpoint WebSocket khác.
BASE = 'https://fapi.binance.com'
WS = os.getenv('BINANCE_WS_URL', 'wss://fstream.binance.com/market/stream?streams=' +
                '/'.join(f'btcusdt@kline_{tf}' for tf in INTERVALS))
# Ánh xạ tên loại dữ liệu nội bộ sang endpoint tương ứng của Binance. Candle
# dùng endpoint klines, còn ratio/taker là các endpoint thống kê futures.
PATHS = {'candle':'/fapi/v1/klines','ratio':'/futures/data/globalLongShortAccountRatio','taker':'/futures/data/takerlongshortRatio',
         'top_ratio':'/futures/data/topLongShortPositionRatio','oi':'/futures/data/openInterestHist',
         'funding':'/fapi/v1/fundingRate'}


async def get(client, path, params):
    # Binance có thể giới hạn tần suất hoặc tạm thời lỗi máy chủ. Hàm này gom
    # toàn bộ chính sách retry để các luồng thu thập không phải tự lặp lại logic.
    for attempt in range(6):
        try:
            response = await client.get(BASE + path, params=params)
            if response.status_code in (418, 429) or response.status_code >= 500:
                # Ưu tiên Retry-After do máy chủ cung cấp; nếu thiếu thì dùng
                # exponential backoff và giới hạn thời gian chờ tối đa 5 phút.
                delay = min(float(response.headers.get('Retry-After', 2 ** attempt)), 300)
                log.warning('REST %s; retry in %.1fs', response.status_code, delay)
                await asyncio.sleep(delay + random.random())
                continue
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.NetworkError):
            # Lỗi timeout hoặc mạng thường có thể tự hồi phục, nên thử lại với
            # thời gian chờ tăng dần. Lần thử cuối cùng được phép phát sinh lỗi.
            if attempt == 5:
                raise
            await asyncio.sleep(2 ** attempt)
    # Nhánh này chỉ xảy ra khi đã dùng hết số lần thử do các mã trạng thái lỗi.
    raise RuntimeError('Binance retry budget exceeded')


async def fetch_range(client, sender, tf, kind, start, end, mode):
    # Lấy dữ liệu trong khoảng [start, end), sau đó chuẩn hóa từng bản ghi và
    # gửi vào Kafka. Binance giới hạn mỗi response tối đa 500 bản ghi nên phải
    # chia khoảng thời gian thành nhiều trang liên tiếp.
    step = INTERVALS[tf]
    # endTime của Binance là inclusive, trong khi khoảng thời gian nội bộ dùng
    # [start, end). Metric chỉ xuất hiện sau khi nến bắt đầu nên cursor khác nhau.
    cursor = start + step if kind in END_STAMPED_KINDS else start
    last = end if kind in END_STAMPED_KINDS else end - 1
    count = 0
    while cursor <= last:
        # Trừ 1 mili-giây để page_end vẫn nằm trong giới hạn trang hiện tại khi
        # endpoint được hiểu là inclusive.
        page_end = min(last, cursor + 500 * step - 1)
        params = {'symbol':'BTCUSDT', 'interval' if kind == 'candle' else 'period':tf,
                  'startTime':cursor, 'endTime':page_end, 'limit':500}
        rows = await get(client, PATHS[kind], params)
        if not isinstance(rows, list):
            raise ValueError(f'unexpected Binance response: {rows}')
        if not rows:
            # Không có dữ liệu vẫn phải tiến cursor để tránh lặp vô hạn ở trang
            # rỗng, đồng thời tiếp tục tìm dữ liệu ở các trang sau.
            cursor = page_end + 1
            continue
        received = now_ms()
        for row in rows:
            # Bỏ các bản ghi ngoài trang đang yêu cầu (nếu API trả dư) trước khi
            # chuyển đổi, nhờ đó không phát sinh dữ liệu trùng hoặc ngoài phạm vi.
            source_ts = int(row[0] if kind == 'candle' else row['timestamp'])
            if not cursor <= source_ts <= page_end:
                continue
            # candle và metric có định dạng JSON khác nhau, nhưng đều được đưa
            # về cùng hợp đồng event trước khi validate và gửi vào Kafka.
            event = candle(tf, row, BASE + PATHS[kind], mode, received) if kind == 'candle' else metric(kind, tf, row, BASE + PATHS[kind], mode, received)
            if event['period_start_ms'] < start or event['period_end_ms'] > end:
                continue
            validate(event)
            sender.send(RAW, INSTRUMENT, event)
            count += 1
        # Đẩy các record đã buffer trong trang hiện tại trước khi chuyển sang
        # trang kế tiếp, giữ độ trễ và lượng bộ nhớ ở mức có thể kiểm soát.
        sender.flush()
        cursor = page_end + 1
        # Nghỉ ngắn để giảm nguy cơ chạm rate limit khi backfill nhiều trang.
        await asyncio.sleep(0.15)
    return count


async def fetch_funding(client, sender, start, end, mode):
    # Funding rate được chốt mỗi 8h; endpoint không có tham số period và trả tối
    # đa 1000 bản ghi. Mỗi bản ghi được gán vào kỳ 1h kết thúc tại giờ chốt.
    count, cursor = 0, start
    while cursor < end:
        rows = await get(client, PATHS['funding'], {'symbol':'BTCUSDT', 'startTime':cursor, 'endTime':end - 1, 'limit':1000})
        if not isinstance(rows, list):
            raise ValueError(f'unexpected Binance response: {rows}')
        if not rows:
            break
        received = now_ms()
        for row in rows:
            event = metric('funding', '1h', row, BASE + PATHS['funding'], mode, received)
            if event['period_start_ms'] < start - INTERVALS['1h'] or event['period_end_ms'] > end:
                continue
            validate(event)
            sender.send(RAW, INSTRUMENT, event)
            count += 1
        sender.flush()
        last = int(rows[-1]['fundingTime'])
        if last + 1 <= cursor:
            break
        cursor = last + 1
        await asyncio.sleep(0.15)
    return count


async def poll_loop(once=False):
    # REST collector đảm nhiệm backfill ban đầu và recovery các khoảng dữ liệu
    # còn thiếu. WebSocket không thay thế bước này vì có thể mất kết nối hoặc
    # chỉ cung cấp candle đã đóng trong thời gian tiến trình đang chạy.
    sender = Sender()
    days = int(os.getenv('BACKFILL_DAYS','7'))
    if not 1 <= days <= 28:
        raise ValueError('BACKFILL_DAYS must be 1..28 (metrics retention)')
    with connect() as db:
        async with httpx.AsyncClient(timeout=25, headers={'User-Agent':'btc-market-pipeline/1.0'}) as client:
            first = True
            while True:
                try:
                    # Dùng thời gian máy chủ Binance để tránh lệch đồng hồ giữa
                    # máy chạy pipeline và nguồn dữ liệu.
                    server = await get(client, '/fapi/v1/time', {})
                    server_now = int(server['serverTime'])
                    counts = {}
                    for tf, step in INTERVALS.items():
                        end = server_now // step * step
                        # Không gọi endpoint metric với period không được Binance
                        # hỗ trợ (đặc biệt là 1m).
                        for kind in COLLECT_KINDS_BY_TIMEFRAME[tf]:
                            if first:
                                # Lần chạy đầu lấy toàn bộ cửa sổ backfill, căn
                                # start theo biên của timeframe hiện tại.
                                start = (server_now - days * 86400_000) // step * step
                            else:
                                # Các lần sau chỉ lấy từ bản ghi mới nhất lùi
                                # thêm hai nến để sửa các khoảng bị trễ hoặc hụt.
                                row = db.execute('SELECT max(period_start_ms) AS latest FROM observations WHERE timeframe=%s AND kind=%s', (tf,kind)).fetchone()
                                latest = row['latest']
                                start = max((server_now - days * 86400_000)//step*step,
                                            (latest - 2*step) if latest is not None else end - days*86400_000)
                            mode = 'backfill' if first else 'recovery'
                            if kind == 'funding':
                                counts[f'{tf}/{kind}'] = await fetch_funding(client, sender, start, end, mode)
                            else:
                                counts[f'{tf}/{kind}'] = await fetch_range(client, sender, tf, kind, start, end, mode)
                    heartbeat(db, 'collector-rest', {'state':'ok','counts':counts,'server_time_ms':server_now})
                    log.info('REST collection complete %s', counts)
                    first = False
                    if once:
                        # Chế độ once phục vụ job backfill/test chạy một lần rồi
                        # kết thúc thay vì giữ tiến trình polling liên tục.
                        return
                except Exception as exc:
                    # Ghi heartbeat lỗi để hệ thống giám sát biết collector đang
                    # gặp sự cố; nếu là chế độ once thì giữ nguyên lỗi cho caller.
                    heartbeat(db, 'collector-rest', {'state':'error','error':str(exc)})
                    log.exception('REST cycle failed; will retry')
                    if once:
                        raise
                # Chu kỳ kế tiếp mặc định cách nhau 60 giây, có thể cấu hình qua
                # POLL_SECONDS mà không cần sửa mã nguồn.
                await asyncio.sleep(int(os.getenv('POLL_SECONDS','60')))


async def websocket_loop():
    # WebSocket cung cấp candle realtime với độ trễ thấp. Vòng lặp ngoài cùng
    # giữ kết nối sống lâu dài và tự kết nối lại khi socket hoặc mạng bị lỗi.
    sender = Sender()
    attempt = 0
    with connect() as db:
        while True:
            try:
                async with ws_connect(WS, ping_interval=20, ping_timeout=30, open_timeout=25) as sock:
                    # Khi kết nối thành công, reset bộ đếm backoff vì các lỗi
                    # về sau nên được tính lại từ đầu.
                    log.info('WebSocket connected: %s', WS)
                    heartbeat(db, 'collector-ws', {'state':'connected','url':WS})
                    attempt = 0
                    last_heartbeat = 0
                    async for message in sock:
                        raw = json.loads(message)
                        event = raw.get('data',raw)
                        k = event.get('k')
                        # Stream có thể chứa nhiều event; chỉ nhận đúng BTCUSDT
                        # và các timeframe đã khai báo trong cấu hình nội bộ.
                        if not k or event.get('s') != 'BTCUSDT' or k.get('i') not in INTERVALS:
                            continue
                        if time.monotonic() - last_heartbeat > 30:
                            # Heartbeat định kỳ xác nhận socket vẫn nhận dữ liệu,
                            # kể cả khi chưa có candle đóng mới.
                            heartbeat(db, 'collector-ws', {'state':'receiving','last_event_ms':event.get('E'), 'last_candle_closed':k['x']})
                            last_heartbeat = time.monotonic()
                        if not k['x']:
                            # Chỉ phát hành candle đã đóng; candle đang hình thành
                            # sẽ còn thay đổi và dễ tạo dữ liệu không nhất quán.
                            continue
                        # Chuyển payload kline của WebSocket về cùng dạng row mà
                        # hàm candle dùng cho dữ liệu REST.
                        row = [k['t'],k['o'],k['h'],k['l'],k['c'],k['v'],k['T'],k['q'],k['n'],k['V'],k['Q']]
                        normalized = candle(k['i'],row,WS,'live')
                        normalized['raw'] = raw
                        validate(normalized)
                        sender.send(RAW, INSTRUMENT, normalized)
                        sender.flush()
                        log.info('WebSocket closed candle %s %s', k['i'], k['t'])
            except Exception as exc:
                # Backoff tăng dần giúp tránh tạo bão kết nối khi Binance hoặc
                # đường truyền đang tạm thời không khả dụng.
                heartbeat(db, 'collector-ws', {'state':'disconnected','error':str(exc)})
                log.exception('WebSocket reconnect')
                await asyncio.sleep(min(60,2 ** min(attempt,6)) + random.random())
                attempt += 1


async def run():
    # Hai collector chạy song song: REST đảm bảo tính đầy đủ của dữ liệu, còn
    # WebSocket đảm bảo dữ liệu mới được nhận gần như ngay khi candle đóng.
    await asyncio.gather(poll_loop(), websocket_loop())
