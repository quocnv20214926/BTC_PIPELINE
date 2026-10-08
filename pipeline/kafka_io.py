import os
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
from .core import RAW, READY, DLQ, MODEL_SIGNALS, canonical


def bootstrap():
    # Cho phép dùng Kafka local mặc định hoặc thay đổi broker qua biến môi
    # trường khi chạy trong Docker/production.
    return os.getenv('KAFKA_BOOTSTRAP', 'localhost:9092')


def init_topics():
    # Tạo topic theo kiểu idempotent: topic đã tồn tại sẽ được bỏ qua để lệnh
    # init có thể chạy lại an toàn sau khi container restart.
    admin = AdminClient({'bootstrap.servers': bootstrap()})
    current = admin.list_topics(timeout=20).topics
    topics = [NewTopic(t, num_partitions=1, replication_factor=1,
                       config={'retention.ms':str(7 * 86400_000)})
              for t in (RAW, READY, DLQ, MODEL_SIGNALS)
              if t not in current]
    if not topics:
        print('all required topics already exist', flush=True)
        return
    for name, result in admin.create_topics(topics).items():
        # result.result() chờ broker xác nhận thật sự, tránh báo init thành công
        # khi request tạo topic vẫn còn đang pending hoặc đã thất bại.
        result.result(30)
        print('created topic', name, flush=True)


class Sender:
    def __init__(self):
        # Idempotence và acks=all ưu tiên không mất message. Callback delivery
        # lưu lỗi để flush() có thể biến lỗi bất đồng bộ thành exception đồng bộ.
        self.producer = Producer({'bootstrap.servers':bootstrap(), 'enable.idempotence':True,
                                  'acks':'all', 'delivery.timeout.ms':30000})
        self.errors = []

    def delivered(self, error, message):
        # Callback được librdkafka gọi sau khi broker xác nhận hoặc từ chối
        # message; chỉ giữ lỗi vì message thành công không cần lưu thêm.
        if error:
            self.errors.append(str(error))

    def send(self, topic, key, value):
        # canonical() tạo payload JSON ổn định trước khi đưa vào Kafka. poll(0)
        # cho producer cơ hội xử lý callback mà không chặn luồng gọi.
        self.producer.produce(topic, key=key, value=canonical(value), on_delivery=self.delivered)
        self.producer.poll(0)

    def flush(self):
        # flush chờ toàn bộ message đang buffer. Cả số message chưa gửi và lỗi
        # delivery đều phải được báo lên để caller không đánh dấu dữ liệu đã xong.
        remaining = self.producer.flush(35)
        if remaining or self.errors:
            errors, self.errors = self.errors, []
            raise RuntimeError(f'Kafka delivery not confirmed: {remaining}; {errors}')
