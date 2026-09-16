CREATE TABLE IF NOT EXISTS raw_events (
    event_id text PRIMARY KEY,
    instrument text NOT NULL,
    timeframe text NOT NULL CHECK (timeframe IN ('1m','5m','15m','1h')),
    kind text NOT NULL CHECK (kind IN ('candle','ratio','taker')),
    period_start_ms bigint NOT NULL,
    received_at_ms bigint NOT NULL,
    envelope jsonb NOT NULL,
    stored_at timestamptz NOT NULL DEFAULT now()
);
-- Nâng cấp schema đã được khởi tạo bởi phiên bản chỉ có 15m/1h.
ALTER TABLE raw_events DROP CONSTRAINT IF EXISTS raw_events_timeframe_check;
ALTER TABLE raw_events ADD CONSTRAINT raw_events_timeframe_check
    CHECK (timeframe IN ('1m','5m','15m','1h'));
CREATE TABLE IF NOT EXISTS observations (
    instrument text NOT NULL,
    timeframe text NOT NULL,
    kind text NOT NULL,
    period_start_ms bigint NOT NULL,
    period_end_ms bigint NOT NULL,
    event_id text NOT NULL REFERENCES raw_events(event_id),
    received_at_ms bigint NOT NULL,
    mode text NOT NULL,
    data jsonb NOT NULL,
    PRIMARY KEY (instrument,timeframe,kind,period_start_ms)
);
CREATE TABLE IF NOT EXISTS feature_sets (
    feature_set_id uuid PRIMARY KEY,
    instrument text NOT NULL,
    timeframe text NOT NULL,
    window_start_ms bigint NOT NULL,
    window_end_ms bigint NOT NULL,
    lookback integer NOT NULL,
    feature_version text NOT NULL,
    max_received_at_ms bigint NOT NULL,
    prepared_at timestamptz NOT NULL DEFAULT now(),
    contains_backfill boolean NOT NULL,
    contains_recovery boolean NOT NULL,
    content_sha256 text NOT NULL,
    payload jsonb NOT NULL,
    UNIQUE(instrument,timeframe,window_end_ms,lookback,feature_version)
);
CREATE TABLE IF NOT EXISTS outbox (
    id bigserial PRIMARY KEY,
    topic text NOT NULL,
    message_key text NOT NULL,
    dedup_key text UNIQUE NOT NULL,
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    published_at timestamptz
);
CREATE INDEX IF NOT EXISTS outbox_pending ON outbox(id) WHERE published_at IS NULL;
CREATE TABLE IF NOT EXISTS processed_offsets (
    topic text NOT NULL,
    partition_id integer NOT NULL,
    offset_id bigint NOT NULL,
    processed_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(topic,partition_id,offset_id)
);
CREATE TABLE IF NOT EXISTS data_issues (
    issue_id text PRIMARY KEY,
    kind text NOT NULL,
    details jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS service_heartbeats (
    service text PRIMARY KEY,
    updated_at timestamptz NOT NULL DEFAULT now(),
    details jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS signal_events (
    signal_id text PRIMARY KEY,
    instrument text NOT NULL,
    signal_type text NOT NULL CHECK (signal_type IN ('model','anomaly')),
    source text NOT NULL,
    timeframe text NOT NULL CHECK (timeframe IN ('1m','5m','15m')),
    event_time_ms bigint NOT NULL,
    produced_at_ms bigint NOT NULL,
    payload jsonb NOT NULL,
    stored_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(source,event_time_ms)
);
CREATE INDEX IF NOT EXISTS signal_events_source_time
    ON signal_events(source,event_time_ms DESC);
CREATE TABLE IF NOT EXISTS trade_decisions (
    decision_id text PRIMARY KEY,
    instrument text NOT NULL,
    event_time_ms bigint NOT NULL,
    produced_at_ms bigint NOT NULL,
    action text NOT NULL CHECK (action IN ('LONG','SHORT','HOLD')),
    trade_allowed boolean NOT NULL,
    payload jsonb NOT NULL,
    stored_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS trade_decisions_time
    ON trade_decisions(event_time_ms DESC);
CREATE TABLE IF NOT EXISTS execution_events (
    execution_id text PRIMARY KEY,
    instrument text NOT NULL,
    event_time_ms bigint NOT NULL,
    event_type text NOT NULL,
    status text NOT NULL,
    payload jsonb NOT NULL,
    stored_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS execution_events_time ON execution_events(event_time_ms DESC);
CREATE TABLE IF NOT EXISTS execution_state (
    instrument text PRIMARY KEY,
    environment text NOT NULL,
    position_side text NOT NULL,
    position_amount numeric NOT NULL DEFAULT 0,
    entry_price numeric NOT NULL DEFAULT 0,
    unrealized_pnl numeric NOT NULL DEFAULT 0,
    stop_loss_price numeric,
    take_profit_price numeric,
    entry_order_id text,
    stop_order_id text,
    take_profit_order_id text,
    protection_status text NOT NULL DEFAULT 'NONE',
    last_sync_at timestamptz NOT NULL DEFAULT now(),
    details jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE OR REPLACE VIEW latest_signals AS
    SELECT DISTINCT ON (source) signal_id,instrument,signal_type,source,timeframe,
           event_time_ms,produced_at_ms,payload,stored_at
    FROM signal_events ORDER BY source,event_time_ms DESC;
CREATE OR REPLACE FUNCTION immutable_feature_set() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'feature_sets are immutable; use a new feature version'; END;
$$;
DROP TRIGGER IF EXISTS feature_sets_immutable ON feature_sets;
CREATE TRIGGER feature_sets_immutable BEFORE UPDATE OR DELETE ON feature_sets
FOR EACH ROW EXECUTE FUNCTION immutable_feature_set();
