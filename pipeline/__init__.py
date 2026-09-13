"""Gói thu thập và xử lý dữ liệu thị trường BTCUSDT.

Pipeline nhận dữ liệu từ Binance, chuẩn hóa thành event bất biến, lưu vào
PostgreSQL, sau đó tạo các feature window và phát thông báo qua Kafka.
"""
