# BTC realtime signal pipeline

Pipeline thu thập Binance USD-M, chuẩn hóa dữ liệu và tạo ba nguồn tín hiệu
độc lập trước khi hợp nhất thành trade intent. Trade intent **không tự đặt lệnh**.

## Luồng dữ liệu

```text
Binance REST/WS
  -> market.raw.v1
  -> PostgreSQL observations + feature_sets
  -> features.ready.v1
       -> M5 model  ---\
       -> M15 model ----> signals.model.v1 ---\

market.raw.v1 -> M1 anomaly -> signals.anomaly.v1
                                             |
                                             v
                                  central decision processor
                                             |
                                             v
                                   decisions.trade.v1
                                             |
                                             v
                          Binance Testnet position manager
                          -> entry + exchange-hosted SL/TP
                          -> executions.testnet.v1

signals.* + decisions.* -> PostgreSQL audit/history/latest state
```

PostgreSQL là source of truth vì cần lịch sử, idempotency và replay/audit.


## Signal contracts

M5/M15 model signal luôn có `probabilities.HOLD/LONG/SHORT`, `predicted_class`,
`confidence`, `risk_ratio`, model version, lookback, horizon và event time.
M1 anomaly có return/range/volume z-score, `anomaly_score`, `is_anomaly` và
severity. Central output có action, trade gate, confidence, ensemble
probabilities, Risk:R, reasons và ID của mọi signal đầu vào.

Model M5/M15 hiện là baseline nhẹ để hoàn thiện pipeline. Có thể thay bằng CNN
qua adapter mà không đổi topic, schema hoặc central processor.

Các artifact runtime nhìn thấy trực tiếp trong project:

- `models/m5/model.json`: model tín hiệu M5 nhẹ, versioned.
- `models/m15/realtime_adapter.json`: adapter realtime M15; checkpoint CNN nằm
  trong `artifacts/m15_final_model` khi chạy ngoài Docker.
- `models/m1/anomaly_detector.json`: detector và threshold bất thường M1.

Worker bắt buộc load các file này khi khởi động; thiếu artifact hoặc sai
timeframe sẽ fail-fast thay vì âm thầm chạy cấu hình mặc định.

## Chạy

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Các service realtime:

- `collector`: REST recovery và closed-candle WebSocket.
- `writer`: validate raw data, lưu observations và tạo feature windows.
- `publisher`: transactional outbox sang Kafka.
- `model-signals`: inference M5/M15.
- `anomaly-signals`: rolling anomaly M1.
- `signal-store`: lưu signal/decision bất biến vào PostgreSQL.
- `central-decision`: hợp nhất tín hiệu, chỉ tạo intent.
- `testnet-executor`: quản lý một vị thế Binance USD-M Futures Testnet, từ chối
  decision cũ/backfill và chỉ mở entry khi có thể tạo đủ SL + TP trên sàn.

### Bật Binance Futures Testnet

Tạo API key riêng tại Binance Futures Testnet, sau đó thêm vào `.env` (không dùng
key tài khoản thật và không commit file này):

```dotenv
TESTNET_EXECUTION_ENABLED=true
BINANCE_TESTNET_API_KEY=your_testnet_key
BINANCE_TESTNET_API_SECRET=your_testnet_secret
TESTNET_ORDER_MARGIN_USDT=100
TESTNET_LEVERAGE=50
TESTNET_STOP_LOSS_PCT=0.006
TESTNET_TAKE_PROFIT_RR=1.5
TESTNET_MAX_DECISION_AGE_SECONDS=90
```

Mặc định executor bị tắt. Khi bật, nó dùng One-way Mode (`positionSide=BOTH`),
chỉ giữ một vị thế BTCUSDT, đặt MARKET entry rồi lập tức đặt `STOP_MARKET` và
`TAKE_PROFIT_MARKET` dạng close-position. Nếu một trong hai protection order lỗi,
entry bị đóng khẩn cấp. Khi restart, vị thế có thật trên testnet là nguồn trạng
thái chính; vị thế thiếu đủ SL/TP cũng bị đóng fail-closed.

`TESTNET_ORDER_MARGIN_USDT=100` và `TESTNET_LEVERAGE=50` tạo exposure danh nghĩa
xấp xỉ 5.000 USDT. Với SL 0,6%, lỗ lý thuyết khoảng 30 USDT trước phí, funding và
slippage. Executor bắt buộc chuyển BTCUSDT sang isolated margin trước khi mở lệnh.

Kiểm tra trạng thái:

```powershell
docker compose --profile tools run --rm cli python -m pipeline status
```

Các bảng chính: `observations`, `feature_sets`, `signal_events`,
`latest_signals`, `trade_decisions`, `service_heartbeats`.

## Quan sát toàn bộ pipeline

Sau khi chạy `docker compose up -d --build`, hai giao diện chỉ được mở trên
máy local:

- Grafana: http://localhost:3000 — dashboard **BTC Pipeline Overview** hiển thị
  tốc độ nhận nến, tín hiệu M1/M5/M15, quyết định LONG/SHORT/HOLD, outbox backlog,
  heartbeat của service và lỗi dữ liệu. Dashboard và PostgreSQL datasource được
  nạp tự động, không cần đăng nhập.
- Kafka UI: http://localhost:8080 — xem topic, message gần nhất, partition,
  consumer group và consumer lag để biết dữ liệu đang dừng ở công đoạn nào.

Chọn khoảng thời gian ở góc phải Grafana (mặc định 6 giờ). Dòng dữ liệu khỏe khi
biểu đồ nến tiếp tục tăng, heartbeat có tuổi nhỏ, outbox pending gần 0 và consumer
lag không tăng liên tục. Các cổng 3000/8080 bind vào `127.0.0.1`, không công khai
ra mạng LAN/Internet.
