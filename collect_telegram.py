"""Thu thập tin nhắn từ kênh Telegram công khai (mặc định @coin369channel) để làm dữ liệu tin tức.

Hai cách:
  1. telethon (khuyến nghị): API chính thức của Telegram, lấy được TOÀN BỘ lịch sử, thời gian chính xác tới giây,
     lượt xem / chuyển tiếp / trả lời, thời điểm sửa tin. Cần api_id / api_hash (tạo ở https://my.telegram.org ->
     API development tools) và đăng nhập bằng số điện thoại MỘT lần (mã xác nhận gửi về app Telegram); phiên đăng
     nhập lưu ở file .session để các lần sau không phải nhập lại.
         pip install telethon
         đặt TG_API_ID, TG_API_HASH trong .env (hoặc biến môi trường)
  2. web: đọc trang xem trước công khai https://t.me/s/<kênh> (không cần tài khoản). Chỉ dùng được nếu kênh bật
     xem trước; số lượt xem làm tròn (1.2K), không có thời điểm sửa tin. Dùng khi chưa có api_id.

Dữ liệu lưu ở historical_data/telegram/<kênh>.jsonl (mỗi dòng 1 tin, UTF-8). Chạy lại sẽ chỉ lấy tin mới
(telethon: id lớn hơn id lớn nhất đã có; web: dừng khi gặp id đã có). Tin bị sửa sau này: telethon ghi edit_date;
khi dùng làm feature phải chỉ dùng tin có date <= thời điểm quyết định, và cẩn thận với nội dung đã sửa sau đó.

Dùng (chạy từ thư mục gốc BTC_PIPELINE):
    python collect_telegram.py --channel coin369channel --method telethon --since 2020-01-01
    python collect_telegram.py --channel coin369channel --method web --since 2024-01-01
    python collect_telegram.py --channel coin369channel --summary          # thống kê dữ liệu đã có

Trong Python:
    import collect_telegram as ct
    df = ct.load_messages("coin369channel")      # DataFrame, cột date là datetime UTC
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

OUT_DIR = Path("historical_data/telegram")
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
FIELDS = ("channel", "msg_id", "date", "ts_ms", "text", "views", "forwards", "replies", "edit_date", "media",
          "urls", "fwd_from", "reply_to", "grouped_id", "method", "collected_at")


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def load_env(path: str | Path = ".env") -> None:
    """Đọc .env đơn giản (KEY=VALUE), không ghi đè biến môi trường đã có."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def out_path(channel: str) -> Path:
    return OUT_DIR / f"{channel.lstrip('@').lower()}.jsonl"


def existing_ids(path: Path) -> set[int]:
    ids = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    ids.add(int(json.loads(line)["msg_id"]))
                except (ValueError, KeyError, json.JSONDecodeError):
                    continue
    return ids


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in FIELDS}, ensure_ascii=False) + "\n")


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_since(s: str | None) -> datetime | None:
    if not s:
        return None
    d = datetime.fromisoformat(s)
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)


def load_messages(channel: str, path: str | Path | None = None):
    """Đọc file jsonl -> DataFrame (mỗi msg_id 1 dòng, bản thu thập mới nhất), sắp theo thời gian."""
    import pandas as pd
    p = Path(path) if path else out_path(channel)
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("collected_at").drop_duplicates("msg_id", keep="last")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["edit_date"] = pd.to_datetime(df["edit_date"], utc=True, errors="coerce")
    return df.sort_values(["date", "msg_id"]).reset_index(drop=True)


def summary(channel: str) -> None:
    import pandas as pd
    df = load_messages(channel)
    if df.empty:
        print("chưa có dữ liệu")
        return
    print(f"{len(df):,} tin, {df.date.min()} -> {df.date.max()}")
    print(f"có chữ: {(df.text.fillna('').str.len() > 0).mean():.1%} | có media: {df.media.notna().mean():.1%} | "
          f"độ dài TB {df.text.fillna('').str.len().mean():.0f} ký tự | bị sửa: {df.edit_date.notna().mean():.1%}")
    m = df.set_index("date").resample("MS").size()
    print("\nSố tin theo tháng (12 tháng gần nhất):")
    print(m.tail(12).to_string())
    h = df.date.dt.tz_convert("Asia/Ho_Chi_Minh").dt.hour.value_counts().sort_index()
    print("\nSố tin theo giờ VN:", dict(h))
    print("\n5 tin gần nhất:")
    for _, r in df.tail(5).iterrows():
        print(f"- [{r.date:%Y-%m-%d %H:%M} UTC] {str(r.text)[:200]!r}")


# --------------------------------------------------------------------------- #
# Cách 1: Telethon (API chính thức)
# --------------------------------------------------------------------------- #
def _media_name(m) -> str | None:
    if m.media is None:
        return None
    return type(m.media).__name__.replace("MessageMedia", "").lower()


def _telethon_row(channel: str, m) -> dict:
    urls = set(URL_RE.findall(m.message or ""))
    for ent, _txt in (m.get_entities_text() if m.entities else []):
        u = getattr(ent, "url", None)
        if u:
            urls.add(u)
    fwd = None
    if m.fwd_from is not None:
        fwd = getattr(m.fwd_from, "from_name", None) or str(getattr(m.fwd_from, "from_id", "") or "") or "forward"
    return {"channel": channel, "msg_id": int(m.id), "date": _iso(m.date), "ts_ms": int(m.date.timestamp() * 1000),
            "text": m.message or "", "views": m.views, "forwards": m.forwards,
            "replies": getattr(m.replies, "replies", None) if m.replies else None, "edit_date": _iso(m.edit_date),
            "media": _media_name(m), "urls": sorted(urls), "fwd_from": fwd,
            "reply_to": getattr(m.reply_to, "reply_to_msg_id", None) if m.reply_to else None,
            "grouped_id": m.grouped_id, "method": "telethon", "collected_at": _now()}


async def _collect_telethon(channel: str, since: datetime | None, api_id: int, api_hash: str, session: str,
                            batch: int = 500) -> int:
    try:
        from telethon import TelegramClient
    except ImportError as exc:
        raise SystemExit("Chưa cài telethon: pip install telethon") from exc
    path = out_path(channel)
    ids = existing_ids(path)
    last = max(ids) if ids else 0
    Path(session).parent.mkdir(parents=True, exist_ok=True)
    n, buf = 0, []
    client = TelegramClient(session, api_id, api_hash, flood_sleep_threshold=600)
    async with client:
        entity = await client.get_entity(channel)
        kw = {"reverse": True}
        if last:
            kw["min_id"] = last                       # chỉ lấy tin mới hơn tin đã có
        elif since:
            kw["offset_date"] = since                 # reverse=True: lấy từ since trở về sau
        print(f"telethon: {getattr(entity, 'title', channel)} | đã có {len(ids):,} tin, id lớn nhất {last}")
        async for m in client.iter_messages(entity, **kw):
            if m.id in ids or m.action is not None:   # bỏ tin hệ thống (đổi tên kênh, ghim...)
                continue
            buf.append(_telethon_row(channel, m))
            if len(buf) >= batch:
                append_rows(path, buf)
                n += len(buf)
                print(f"  +{n:,} tin (tới {buf[-1]['date']})", flush=True)
                buf = []
    append_rows(path, buf)
    n += len(buf)
    return n


def collect_telethon(channel: str, since: datetime | None) -> int:
    load_env()
    api_id, api_hash = os.environ.get("TG_API_ID"), os.environ.get("TG_API_HASH")
    if not api_id or not api_hash:
        raise SystemExit("Thiếu TG_API_ID / TG_API_HASH (tạo ở https://my.telegram.org, đặt trong .env)")
    session = os.environ.get("TG_SESSION", str(OUT_DIR / "tg_collector"))
    return asyncio.run(_collect_telethon(channel, since, int(api_id), api_hash, session))


# --------------------------------------------------------------------------- #
# Cách 2: trang xem trước công khai t.me/s/<kênh>
# --------------------------------------------------------------------------- #
class _PreviewParser(HTMLParser):
    """Đọc các tin trong HTML của https://t.me/s/<kênh>. Bỏ qua phần trích dẫn tin được trả lời."""

    VOID = {"br", "img", "meta", "link", "input", "hr", "source", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.msgs: list[dict] = []
        self.stack: list[tuple[str, set]] = []
        self.cur: dict | None = None
        self.cur_depth = -1
        self.text_depth = -1
        self.views_depth = -1
        self.reply_depth = -1
        self.fwd_depth = -1

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = set((a.get("class") or "").split())
        if tag not in self.VOID:
            self.stack.append((tag, cls))
        d = len(self.stack)
        if tag == "div" and "tgme_widget_message" in cls and a.get("data-post"):
            post = a["data-post"]
            self.cur = {"post": post, "msg_id": int(post.rsplit("/", 1)[-1]), "text": [], "urls": set(), "date": None,
                        "views": None, "media": None, "fwd_from": None, "reply_to": None}
            self.cur_depth = d
            return
        if self.cur is None:
            return
        if "tgme_widget_message_reply" in cls and self.reply_depth < 0:
            self.reply_depth = d
            href = a.get("href", "")
            if href:
                try:
                    self.cur["reply_to"] = int(href.rstrip("/").rsplit("/", 1)[-1].split("?")[0])
                except ValueError:
                    pass
        if self.reply_depth >= 0:
            return
        if "tgme_widget_message_text" in cls and self.text_depth < 0:
            self.text_depth = d
        elif "tgme_widget_message_views" in cls:
            self.views_depth = d
        elif "tgme_widget_message_forwarded_from" in cls or "tgme_widget_message_forwarded_from_name" in cls:
            self.fwd_depth = d if self.fwd_depth < 0 else self.fwd_depth
            self.cur["fwd_from"] = self.cur["fwd_from"] or ""
        if tag == "time" and a.get("datetime") and self.cur["date"] is None:
            self.cur["date"] = a["datetime"]
        for key, name in (("tgme_widget_message_photo_wrap", "photo"), ("tgme_widget_message_video_player", "video"),
                          ("tgme_widget_message_document", "document"), ("tgme_widget_message_poll", "poll"),
                          ("tgme_widget_message_sticker_wrap", "sticker"), ("tgme_widget_message_voice", "voice")):
            if key in cls and self.cur["media"] is None:
                self.cur["media"] = name
        if self.text_depth >= 0:
            if tag == "br":
                self.cur["text"].append("\n")
            if tag == "a" and a.get("href"):
                self.cur["urls"].add(a["href"])

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        while self.stack:
            t, _ = self.stack.pop()
            d = len(self.stack) + 1
            if d == self.text_depth:
                self.text_depth = -1
            if d == self.views_depth:
                self.views_depth = -1
            if d == self.reply_depth:
                self.reply_depth = -1
            if d == self.fwd_depth:
                self.fwd_depth = -1
            if d == self.cur_depth and self.cur is not None:
                self._close()
            if t == tag:
                break

    def handle_data(self, data):
        if self.cur is None or self.reply_depth >= 0:
            return
        if self.text_depth >= 0:
            self.cur["text"].append(data)
        elif self.views_depth >= 0 and data.strip():
            self.cur["views"] = data.strip()
        elif self.fwd_depth >= 0 and data.strip():
            self.cur["fwd_from"] = (self.cur["fwd_from"] or "") + data.strip()

    def _close(self):
        m = self.cur
        self.cur, self.cur_depth, self.text_depth, self.views_depth, self.reply_depth, self.fwd_depth = None, -1, -1, -1, -1, -1
        m["text"] = re.sub(r"[ \t]+\n", "\n", "".join(m["text"])).strip()
        self.msgs.append(m)


def _views(s: str | None) -> int | None:
    if not s:
        return None
    s = s.strip().upper().replace(",", "")
    mult = {"K": 1e3, "M": 1e6}.get(s[-1:], 1)
    try:
        return int(float(s.rstrip("KM")) * mult)
    except ValueError:
        return None


def parse_preview(html: str, channel: str) -> list[dict]:
    p = _PreviewParser()
    p.feed(html)
    rows = []
    for m in p.msgs:
        if not m["date"]:
            continue
        dt = datetime.fromisoformat(m["date"].replace("Z", "+00:00"))
        rows.append({"channel": channel, "msg_id": m["msg_id"], "date": _iso(dt), "ts_ms": int(dt.timestamp() * 1000),
                     "text": m["text"], "views": _views(m["views"]), "forwards": None, "replies": None, "edit_date": None,
                     "media": m["media"], "urls": sorted(m["urls"] | set(URL_RE.findall(m["text"]))),
                     "fwd_from": re.sub(r"^Forwarded from\s*", "", m["fwd_from"]) if m["fwd_from"] is not None else None,
                     "reply_to": m["reply_to"], "grouped_id": None, "method": "web",
                     "collected_at": _now()})
    return rows


def _get(url: str, retries: int = 4) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                                               "Accept-Language": "en-US,en;q=0.8"})
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as e:
            wait = 5 * (k + 1) * (6 if getattr(e, "code", None) == 429 else 1)
            print(f"  lỗi {e} -> thử lại sau {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"không tải được {url}")


def collect_web(channel: str, since: datetime | None, pause: float = 1.5, max_pages: int = 100000) -> int:
    ch = channel.lstrip("@")
    path = out_path(ch)
    ids = existing_ids(path)
    before, n, pages = None, 0, 0
    while pages < max_pages:
        url = f"https://t.me/s/{ch}" + (f"?before={before}" if before else "")
        rows = parse_preview(_get(url), ch)
        pages += 1
        if not rows:
            if pages == 1:
                raise SystemExit("không đọc được tin nào (kênh không tồn tại / tắt xem trước / HTML đổi định dạng)")
            break
        new = [r for r in rows if r["msg_id"] not in ids and (since is None or r["date"] >= since.isoformat())]
        append_rows(path, new)
        ids.update(r["msg_id"] for r in new)
        n += len(new)
        oldest = min(rows, key=lambda r: r["msg_id"])
        print(f"  trang {pages}: {len(rows)} tin, mới {len(new)}, cũ nhất #{oldest['msg_id']} {oldest['date']}", flush=True)
        if since is not None and oldest["date"] < since.isoformat():
            break                            # đã lùi tới trước mốc since
        if oldest["msg_id"] <= 1:
            break                            # tới tin đầu tiên của kênh
        if not new and ids:
            break                            # cả trang đã có -> đã nối vào phần thu lần trước
        before = oldest["msg_id"]
        time.sleep(pause)
    return n


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Thu thập tin nhắn kênh Telegram công khai")
    ap.add_argument("--channel", default="coin369channel")
    ap.add_argument("--method", choices=("telethon", "web"), default="telethon")
    ap.add_argument("--since", default="2020-01-01", help="chỉ lấy tin từ ngày này (UTC), lần chạy đầu")
    ap.add_argument("--summary", action="store_true", help="chỉ in thống kê dữ liệu đã thu")
    a = ap.parse_args()
    ch = a.channel.lstrip("@")
    if a.summary:
        summary(ch)
        return
    since = parse_since(a.since)
    t0 = time.time()
    n = collect_telethon(ch, since) if a.method == "telethon" else collect_web(ch, since)
    print(f"xong: +{n:,} tin mới -> {out_path(ch)} ({time.time() - t0:.0f}s)")
    summary(ch)


if __name__ == "__main__":
    sys.exit(main())
