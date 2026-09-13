import json
import logging
import os
import time
from decimal import InvalidOperation

from confluent_kafka import Consumer
from psycopg.types.json import Jsonb

from .core import RAW, DLQ, INSTRUMENT, validate
from .kafka_io import Sender, bootstrap
from .storage import connect, save_event, build_windows, heartbeat, issue

log = logging.getLogger(__name__)


def writer(stop_after=None):
    # Writer là consumer duy nhất chịu trách nhiệm biến raw Kafka event thành
    # dữ liệu PostgreSQL đã validate và tạo feature window mới.
    consumer = Consumer({'bootstrap.servers':bootstrap(),'group.id':'market-writer',
        'auto.offset.reset':'earliest','enable.auto.commit':False,'max.poll.interval.ms':300000})
    consumer.subscribe([RAW])
    started = time.monotonic()
    try:
        with connect() as db:
            while stop_after is None or time.monotonic() - started < stop_after:
                # Consume theo batch giảm overhead transaction nhưng vẫn giới
                # hạn số message để thời gian xử lý không vượt poll interval.
                messages = consumer.consume(num_messages=200,timeout=1)
                if not messages:
                    # Heartbeat idle giúp phân biệt worker đang sống với worker
                    # bị treo trong khi Kafka tạm thời không có message.
                    heartbeat(db,'writer',{'state':'idle'})
                    continue
                for m in messages:
                    if m.error():
                        # Lỗi transport của Kafka không nên bị coi là dữ liệu
                        # xấu; ném exception để process dừng và được khởi động lại.
                        raise RuntimeError(str(m.error()))
                inserted = 0
                with db.transaction():
                    for m in messages:
                        # Offset được ghi trước khi xử lý để chống xử lý lặp khi
                        # consumer restart hoặc commit offset chưa kịp hoàn tất.
                        offset_key = (m.topic(),m.partition(),m.offset())
                        fresh = db.execute('INSERT INTO processed_offsets(topic,partition_id,offset_id) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING RETURNING offset_id', offset_key).fetchone()
                        if not fresh:
                            continue
                        try:
                            # Parse và validate trước khi chạm vào bảng nghiệp vụ;
                            # message hỏng được chuyển sang issue và DLQ.
                            e = json.loads(m.value())
                            validate(e)
                        except (ValueError,KeyError,TypeError,InvalidOperation) as exc:
                            # DLQ giữ payload/error đủ để điều tra mà không làm
                            # hỏng cả batch hợp lệ còn lại.
                            key = ':'.join(map(str,offset_key))
                            details = {'source_topic':m.topic(),'partition':m.partition(),'offset':m.offset(),
                                       'error':str(exc),'raw':m.value().decode('utf-8',errors='replace')}
                            issue(db,key,'invalid_message',details)
                            db.execute('INSERT INTO outbox(topic,message_key,dedup_key,payload) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                                       (DLQ,INSTRUMENT,'dlq:'+key,Jsonb(details)))
                            continue
                        inserted += save_event(db,e)
                    # Chỉ rebuild window khi batch có observation mới; điều này
                    # tránh quét lại toàn bộ dữ liệu khi batch chỉ chứa duplicate.
                    windows = build_windows(db,int(os.getenv('LOOKBACK','60'))) if inserted else 0
                    heartbeat(db,'writer',{'state':'ok','batch':len(messages),'inserted':inserted,'feature_sets_created':windows})
                # DB transaction đã thành công; nếu crash trước commit offset thì
                # batch sẽ replay nhưng processed_offsets/save_event sẽ idempotent.
                consumer.commit(asynchronous=False)
                log.info('writer messages=%s inserted=%s windows=%s',len(messages),inserted,windows)
    finally:
        consumer.close()


def publish(once=False):
    # Publisher đọc transactional outbox thay vì gửi trực tiếp trong writer.
    # Đây là ranh giới giúp DB commit và việc phát Kafka có thể retry độc lập.
    sender = Sender()
    with connect() as db:
        while True:
            with db.transaction():
                # SKIP LOCKED cho phép nhiều publisher cùng chạy mà không chờ
                # nhau trên các row đã được worker khác giữ lock.
                rows = db.execute('SELECT * FROM outbox WHERE published_at IS NULL ORDER BY id LIMIT 100 FOR UPDATE SKIP LOCKED').fetchall()
                for row in rows:
                    sender.send(row['topic'],row['message_key'],row['payload'])
                sender.flush()
                for row in rows:
                    # Chỉ đánh dấu published sau flush thành công; nếu flush lỗi
                    # transaction rollback và lần chạy sau sẽ thử lại.
                    db.execute('UPDATE outbox SET published_at=now() WHERE id=%s',(row['id'],))
                heartbeat(db,'publisher',{'state':'ok','published_batch':len(rows)})
            if rows:
                log.info('published %s outbox messages',len(rows))
            if once and not rows:
                # Chế độ once kết thúc khi queue đã rỗng, phù hợp cho job batch.
                return
            # Poll ngắn để giảm độ trễ mà không tạo vòng lặp bận khi outbox rỗng.
            time.sleep(1)
