import hashlib
import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .core import INSTRUMENT, INTERVALS, KINDS_BY_TIMEFRAME, VERSION, READY, canonical, complete_windows, window_id


def connect():
    # Dùng dict_row để các hàm truy cập cột bằng tên thay vì phụ thuộc thứ tự
    # SELECT. autocommit phù hợp với các thao tác heartbeat/issue đơn lẻ.
    return psycopg.connect(os.environ['DATABASE_URL'], autocommit=True, row_factory=dict_row)


def init_schema():
    # Schema được quản lý tập trung trong sql/schema.sql để lệnh init và Docker
    # sử dụng cùng một định nghĩa bảng.
    with connect() as db:
        db.execute(Path('sql/schema.sql').read_text())


def heartbeat(db, service, details):
    # Mỗi service có đúng một heartbeat hiện tại; upsert giúp cập nhật trạng
    # thái mà không tạo vô hạn dòng lịch sử trong bảng giám sát.
    db.execute('INSERT INTO service_heartbeats VALUES (%s,now(),%s) ON CONFLICT(service) DO UPDATE SET updated_at=now(),details=excluded.details', (service, Jsonb(details)))


def issue(db, key, kind, details):
    # Issue dùng key làm định danh chống trùng, nên cùng một lỗi được gặp lại
    # khi replay Kafka cũng không tạo thêm bản ghi lỗi.
    db.execute('INSERT INTO data_issues(issue_id,kind,details) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING', (key, kind, Jsonb(details)))


def save_event(db, e):
    # raw_events lưu toàn bộ envelope để audit/replay; observations lưu phần
    # nghiệp vụ đã chuẩn hóa và dùng conflict handling cho tính idempotent.
    inserted = db.execute('''INSERT INTO raw_events(event_id,instrument,timeframe,kind,period_start_ms,received_at_ms,envelope)
        VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING event_id''',
        (e['event_id'], e['instrument'], e['timeframe'], e['kind'], e['period_start_ms'], e['received_at_ms'], Jsonb(e))).fetchone()
    if not inserted:
        # Event đã tồn tại trong raw log, thường do producer retry hoặc consumer
        # replay, nên không cần ghi lại observation.
        return False
    row = db.execute('''INSERT INTO observations VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT DO NOTHING RETURNING event_id''',
        (e['instrument'], e['timeframe'], e['kind'], e['period_start_ms'], e['period_end_ms'], e['event_id'], e['received_at_ms'], e['mode'], Jsonb(e['data']))).fetchone()
    if not row:
        # Một period đã có event khác được chấp nhận. Giữ raw revision để audit,
        # nhưng không thay thế observation bất biến đầu tiên.
        issue(db, e['event_id'], 'source_revision', {'message':'First accepted observation retained; revision preserved in raw_events', 'event_id':e['event_id']})
    return bool(row)


def build_windows(db, lookback):
    # Mỗi timeframe chỉ join các nguồn mà Binance thực sự hỗ trợ. 1m dùng
    # candle; từ 5m trở lên dùng candle + ratio + taker như pipeline cũ.
    if lookback < 2:
        raise ValueError('LOOKBACK must be at least 2')
    created = 0
    for tf, step in INTERVALS.items():
        # SQL đã sắp xếp theo thời gian; complete_windows tiếp tục loại các
        # đoạn bị đứt timestamp trước khi tính feature set.
        kinds = KINDS_BY_TIMEFRAME[tf]
        if kinds == ('candle',):
            rows = db.execute('''SELECT c.period_start_ms,
                jsonb_build_object('event_id',c.event_id,'data',c.data,'received_at_ms',c.received_at_ms,'mode',c.mode) AS candle
                FROM observations c
                WHERE c.instrument=%s AND c.timeframe=%s AND c.kind='candle'
                ORDER BY c.period_start_ms''', (INSTRUMENT, tf)).fetchall()
        else:
            rows = db.execute('''SELECT c.period_start_ms,
                jsonb_build_object('event_id',c.event_id,'data',c.data,'received_at_ms',c.received_at_ms,'mode',c.mode) AS candle,
                jsonb_build_object('event_id',r.event_id,'data',r.data,'received_at_ms',r.received_at_ms,'mode',r.mode) AS ratio,
                jsonb_build_object('event_id',t.event_id,'data',t.data,'received_at_ms',t.received_at_ms,'mode',t.mode) AS taker
                FROM observations c
                JOIN observations r USING(instrument,timeframe,period_start_ms)
                JOIN observations t USING(instrument,timeframe,period_start_ms)
                WHERE c.instrument=%s AND c.timeframe=%s AND c.kind='candle' AND r.kind='ratio' AND t.kind='taker'
                ORDER BY c.period_start_ms''', (INSTRUMENT, tf)).fetchall()
        existing = {r['window_end_ms'] for r in db.execute('SELECT window_end_ms FROM feature_sets WHERE timeframe=%s AND lookback=%s AND feature_version=%s', (tf, lookback, VERSION))}
        for window in complete_windows(rows, tf, lookback):
            # end là mốc ngay sau period cuối, dùng làm định danh và biên phải
            # của window theo quy ước [window_start, window_end).
            end = window[-1]['period_start_ms'] + step
            if end in existing:
                continue
            fid = window_id(tf, end, lookback)
            points = [r[k] for r in window for k in kinds]
            latest = max(p['received_at_ms'] for p in points)
            # Gắn provenance tổng hợp để downstream biết window có phụ thuộc dữ
            # liệu backfill/recovery hay chỉ gồm dữ liệu live.
            backfill = any(p['mode'] == 'backfill' for p in points)
            recovery = any(p['mode'] == 'recovery' for p in points)
            payload = {'instrument':INSTRUMENT, 'timeframe':tf, 'window_end_ms':end,
                       'feature_version':VERSION, 'lookback':lookback, 'rows':window,
                       'available_kinds':list(kinds),
                       'availability_policy':'observed-at-ingestion; historical publication times unknown'}
            digest = hashlib.sha256(canonical(payload).encode()).hexdigest()
            # Ghi feature set và outbox trong cùng transaction của writer. Nhờ
            # vậy không có thông báo ready nếu feature set chưa commit thành công.
            record = db.execute('''INSERT INTO feature_sets(feature_set_id,instrument,timeframe,window_start_ms,window_end_ms,
                lookback,feature_version,max_received_at_ms,contains_backfill,contains_recovery,content_sha256,payload)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING feature_set_id''',
                (fid, INSTRUMENT, tf, window[0]['period_start_ms'], end, lookback, VERSION, latest, backfill, recovery, digest, Jsonb(payload))).fetchone()
            if record:
                notice = {'event_type':'feature_set.ready', 'feature_set_id':fid, 'instrument':INSTRUMENT,
                          'timeframe':tf,'window_end_ms':end,'lookback':lookback,'feature_version':VERSION,
                          'contains_backfill':backfill,'contains_recovery':recovery,'max_received_at_ms':latest}
                db.execute('INSERT INTO outbox(topic,message_key,dedup_key,payload) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                           (READY, INSTRUMENT, fid, Jsonb(notice)))
                created += 1
    return created
