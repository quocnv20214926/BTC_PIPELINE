CREATE TABLE IF NOT EXISTS raw_events (
    event_id text PRIMARY KEY,
    instrument text NOT NULL,
    timeframe text NOT NULL CHECK (timeframe IN ('15m','1h')),
    kind text NOT NULL CHECK (kind IN ('candle','ratio','taker')),
    period_start_ms bigint NOT NULL,
    received_at_ms bigint NOT NULL,
    envelope jsonb NOT NULL,
    stored_at timestamptz NOT NULL DEFAULT now()
);
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
CREATE OR REPLACE FUNCTION immutable_feature_set() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'feature_sets are immutable; use a new feature version'; END;
$$;
DROP TRIGGER IF EXISTS feature_sets_immutable ON feature_sets;
CREATE TRIGGER feature_sets_immutable BEFORE UPDATE OR DELETE ON feature_sets
FOR EACH ROW EXECUTE FUNCTION immutable_feature_set();
