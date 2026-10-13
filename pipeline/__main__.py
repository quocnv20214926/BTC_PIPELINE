import argparse
import asyncio
import json
import logging
from pathlib import Path

from .storage import connect, init_schema
from .kafka_io import init_topics
from .core import INTERVALS


def report():
    # Báo cáo được lấy trực tiếp từ các bảng trạng thái chính để phản ánh số
    # lượng dữ liệu đã nhận, feature đã tạo, message đang chờ publish và lỗi.
    with connect() as db:
        return {
            'observations': db.execute('SELECT timeframe,kind,count(*) AS count,min(period_start_ms) AS first_start_ms,max(period_end_ms) AS last_end_ms FROM observations GROUP BY timeframe,kind ORDER BY timeframe,kind').fetchall(),
            'feature_sets': db.execute('SELECT timeframe,count(*) AS count,min(window_end_ms) AS first_end_ms,max(window_end_ms) AS last_end_ms FROM feature_sets GROUP BY timeframe ORDER BY timeframe').fetchall(),
            'outbox': db.execute('SELECT count(*) AS total,count(*) FILTER(WHERE published_at IS NULL) AS pending FROM outbox').fetchone(),
            'issues': db.execute('SELECT kind,count(*) AS count FROM data_issues GROUP BY kind').fetchall(),
            'services': db.execute('SELECT * FROM service_heartbeats ORDER BY service').fetchall(),
            'latest_signals': db.execute("SELECT source,timeframe,event_time_ms,payload->'decision' AS decision FROM latest_signals ORDER BY source").fetchall(),
            'telegram': db.execute("SELECT kind,count(*) AS count,max(sent_at) AS last_sent FROM telegram_messages GROUP BY kind ORDER BY kind").fetchall(),
            'strategy_state': db.execute('SELECT * FROM strategy_state ORDER BY source').fetchall(),
            'strategy_trades': db.execute('''SELECT source,status,count(*) AS count,round(sum(r_net)::numeric,3) AS sum_r_net
                FROM strategy_trades GROUP BY source,status ORDER BY source,status''').fetchall(),
        }


def export():
    # Tạo các file JSON nhỏ phục vụ việc kiểm tra nhanh ngoài database hoặc
    # dùng làm artifact trong môi trường chạy pipeline tự động.
    folder = Path('artifacts')
    folder.mkdir(exist_ok=True)
    (folder/'status.json').write_text(json.dumps(report(),indent=2,default=str),encoding='utf-8')
    with connect() as db:
        for tf in INTERVALS:
            # Chỉ xuất feature mới nhất cho từng timeframe để file artifact
            # không phình to theo toàn bộ lịch sử dữ liệu.
            row = db.execute('SELECT * FROM feature_sets WHERE timeframe=%s ORDER BY window_end_ms DESC LIMIT 1',(tf,)).fetchone()
            if row:
                (folder/f'feature_set_{tf}.json').write_text(json.dumps(row,indent=2,default=str),encoding='utf-8')
    print('Exported artifacts/status.json and latest feature_set JSON per timeframe')


def main():
    # CLI là điểm vào duy nhất khi chạy `python -m pipeline`. Import worker
    # bên trong từng nhánh giúp lệnh status/export không cần tải mọi dependency
    # của collector hoặc Kafka ngay từ đầu.
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s')
    # httpx log cả URL ở mức INFO; URL Telegram chứa token bot -> không log.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['init','collect','backfill','write','publish','strategy-h1',
                                           'check-positioning','public-site','telegram','telegram-setup','status','export'])
    parser.add_argument('--day', help='check-positioning: ngày YYYY-MM-DD (mặc định: 2 ngày trước)')
    args = parser.parse_args()
    if args.command == 'init':
        # Khởi tạo schema PostgreSQL trước, sau đó tạo các topic Kafka cần thiết.
        init_schema()
        init_topics()
    elif args.command in ('collect','backfill'):
        # collect chạy liên tục cả REST lẫn WebSocket; backfill chỉ chạy một
        # chu kỳ REST rồi thoát, phù hợp cho job định kỳ hoặc kiểm tra thủ công.
        from .collector import run,poll_loop
        asyncio.run(run() if args.command=='collect' else poll_loop(once=True))
    elif args.command == 'write':
        # Writer đọc raw event từ Kafka và ghi transactionally vào PostgreSQL.
        from .workers import writer
        writer()
    elif args.command == 'publish':
        # Publisher phát các thông báo đã được ghi vào outbox sau khi feature
        # set đã commit thành công trong database.
        from .workers import publish
        publish()
    elif args.command == 'strategy-h1':
        # Chiến lược H1 (touch + hướng B): tín hiệu mỗi giờ + bộ thực thi lệnh paper (strategy_trades).
        from .strategy_live import strategy_worker
        strategy_worker()
    elif args.command == 'check-positioning':
        # So timestamp REST của dữ liệu vị thế với data.binance.vision (quy ước dùng khi train).
        from .strategy_live import check_positioning
        check_positioning(args.day)
    elif args.command == 'public-site':
        # Trang công khai (% tài khoản): ghi artifacts/public + đẩy lên GitHub nếu cấu hình.
        from .public_site import public_worker
        public_worker()
    elif args.command == 'telegram':
        # Thông báo lệnh paper của strategy-h1 (mở / đóng / bỏ qua) qua Telegram bot.
        from .telegram_bot import telegram_worker
        telegram_worker()
    elif args.command == 'telegram-setup':
        # Kiểm tra token, in chat id, gửi tin thử.
        from .telegram_bot import telegram_setup
        telegram_setup()
    elif args.command == 'status':
        # In cùng nội dung với report() ra stdout để phù hợp với shell/monitoring.
        print(json.dumps(report(),indent=2,default=str))
    elif args.command == 'export':
        export()


if __name__ == '__main__':
    main()
