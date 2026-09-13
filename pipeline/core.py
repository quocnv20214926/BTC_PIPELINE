import hashlib
import json
import time
import uuid
from decimal import Decimal, InvalidOperation

# Độ dài timeframe tính bằng mili-giây. Mọi timestamp trong pipeline đều dùng
# cùng đơn vị này để tránh trộn lẫn giây và mili-giây khi căn chỉnh dữ liệu.
INTERVALS = {'15m': 900_000, '1h': 3_600_000}
# Các hằng số này là tên định danh dùng chung giữa producer, consumer và DB.
INSTRUMENT = 'binance:usdm:BTCUSDT'
RAW = 'market.raw.v1'
READY = 'features.ready.v1'
DLQ = 'market.dlq.v1'
# VERSION được đưa vào event/window ID để một thay đổi schema tạo ra định danh
# mới thay vì âm thầm ghi đè dữ liệu được tạo bởi phiên bản cũ.
VERSION = 'raw-window-v1'


def now_ms():
    # time.time_ns() cho độ phân giải cao hơn; chia cho một triệu để trả về
    # Unix timestamp tính bằng mili-giây giống Binance và PostgreSQL schema.
    return time.time_ns() // 1_000_000


def canonical(value):
    # Biểu diễn JSON ổn định là đầu vào của hash. Sort key và separator cố định
    # giúp cùng một nội dung luôn tạo ra cùng một chuỗi dù thứ tự dict ban đầu.
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def number(value):
    # Decimal tránh sai số float khi xử lý giá, khối lượng và tỷ lệ. Chuỗi sau
    # normalize được lưu trong event để hash và kết quả giữa các nguồn nhất quán.
    n = Decimal(str(value))
    if not n.is_finite():
        raise ValueError('non-finite number')
    return format(n.normalize(), 'f')


def envelope(kind, tf, start, data, source, mode, raw, received=None):
    # Envelope là hợp đồng chung của mọi event. event_id chỉ phụ thuộc dữ liệu
    # nghiệp vụ, nên REST retry và WebSocket trùng bản ghi vẫn được deduplicate.
    received = received or now_ms()
    identity = [INSTRUMENT, kind, tf, start, data]
    return {'schema_version': 1, 'event_id': hashlib.sha256(canonical(identity).encode()).hexdigest(),
            'instrument': INSTRUMENT, 'kind': kind, 'timeframe': tf,
            'period_start_ms': start, 'period_end_ms': start + INTERVALS[tf],
            'received_at_ms': received, 'source': source, 'mode': mode,
            'data': data, 'raw': raw}


def candle(tf, row, source, mode, received=None):
    # Chuyển mảng kline của Binance thành object có tên trường rõ ràng. Binance
    # trả close time inclusive, nên phép cộng 1 ở đây kiểm tra đúng độ dài nến.
    start = int(row[0])
    if int(row[6]) + 1 != start + INTERVALS[tf]:
        raise ValueError('unexpected candle close time')
    fields = ['open', 'high', 'low', 'close', 'volume']
    data = {name: number(value) for name, value in zip(fields, row[1:6])}
    data.update(quote_volume=number(row[7]), trade_count=int(row[8]),
                taker_buy_base=number(row[9]), taker_buy_quote=number(row[10]))
    return envelope('candle', tf, start, data, source, mode, row, received)


def metric(kind, tf, row, source, mode, received=None):
    # Ratio và taker dùng dict JSON thay vì mảng kline. Riêng ratio có timestamp
    # là thời điểm kết thúc, còn taker dùng thời điểm bắt đầu của kỳ.
    ts = int(row['timestamp'])
    # Binance documentation: global account ratio timestamp=end; taker timestamp=start.
    start = ts - INTERVALS[tf] if kind == 'ratio' else ts
    fields = ['longShortRatio', 'longAccount', 'shortAccount'] if kind == 'ratio' else ['buySellRatio', 'buyVol', 'sellVol']
    data = {name: number(row[name]) for name in fields}
    data['source_timestamp_ms'] = ts
    return envelope(kind, tf, start, data, source, mode, row, received)


def validate(e):
    # Kiểm tra event trước khi ghi hoặc phát hành: schema, timeframe, timestamp,
    # provenance, giá trị số và event hash đều phải hợp lệ.
    if e['schema_version'] != 1 or e['instrument'] != INSTRUMENT:
        raise ValueError('unsupported schema or instrument')
    tf = e['timeframe']
    step = INTERVALS[tf]
    start, end = e['period_start_ms'], e['period_end_ms']
    if start % step or end != start + step:
        raise ValueError('unaligned period')
    if end > e['received_at_ms'] or e['received_at_ms'] > now_ms() + 60_000:
        raise ValueError('open/future period or invalid receipt time')
    if e['mode'] not in ('backfill', 'recovery', 'live'):
        raise ValueError('invalid provenance mode')
    d = e['data']
    if e['kind'] == 'candle':
        # OHLCV phải hữu hạn, dương hợp lý và thỏa quan hệ hình học của nến:
        # low không cao hơn open/close, high không thấp hơn open/close.
        o, h, l, c, v = (Decimal(d[k]) for k in ('open', 'high', 'low', 'close', 'volume'))
        if not all(x.is_finite() for x in (o, h, l, c, v)):
            raise ValueError('invalid candle number')
        if min(o, h, l, c) <= 0 or v < 0 or l > min(o, c) or h < max(o, c) or l > h:
            raise ValueError('invalid OHLCV')
    elif e['kind'] in ('ratio', 'taker'):
        # Các metric là tỷ lệ/khối lượng nên không được âm; longAccount và
        # shortAccount phải nằm trong [0,1] và tổng xấp xỉ 1.
        keys = ('longShortRatio', 'longAccount', 'shortAccount') if e['kind'] == 'ratio' else ('buySellRatio', 'buyVol', 'sellVol')
        vals = [Decimal(d[k]) for k in keys]
        if any(not x.is_finite() or x < 0 for x in vals):
            raise ValueError('invalid metric')
        if e['kind'] == 'ratio' and (vals[1] > 1 or vals[2] > 1 or abs(vals[1] + vals[2] - 1) > Decimal('0.001')):
            raise ValueError('invalid account fractions')
    else:
        raise ValueError('unknown event type')
    expected = hashlib.sha256(canonical([INSTRUMENT, e['kind'], tf, start, d]).encode()).hexdigest()
    if expected != e['event_id']:
        raise ValueError('event hash mismatch')


def window_id(tf, end, lookback):
    # UUID ổn định giúp cùng một timeframe/end/lookback luôn trỏ tới một
    # feature set, kể cả khi pipeline được restart hoặc chạy lại backfill.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f'{INSTRUMENT}/{tf}/{end}/{lookback}/{VERSION}'))


def complete_windows(rows, tf, lookback):
    """Sinh các cửa sổ liên tục đủ candle, ratio và taker.

    ``rows`` phải được sắp xếp theo thời gian và mỗi phần tử đã được join đủ
    ba loại dữ liệu cho một period. Khoảng bị thiếu một timeframe sẽ làm cửa sổ
    tương ứng không được phát hành.
    """
    for i in range(lookback - 1, len(rows)):
        window = rows[i - lookback + 1:i + 1]
        starts = [r['period_start_ms'] for r in window]
        if all(b - a == INTERVALS[tf] for a, b in zip(starts, starts[1:])):
            yield window
