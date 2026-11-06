"""Làm sạch tin Telegram (kênh Tin nhanh - 369) -> bảng MỤC TIN (1 dòng = 1 tin đơn lẻ).

Định dạng của kênh (thay đổi theo thời gian):
  - "Tin nhanh:" (2023): nhiều mục "* HH:MM: nội dung" trong 1 tin            -> tách từng mục
  - 🔴 / 🔹 / 🔵 "HH:MM: nội dung": 1 mục (🔴 thường là số liệu / thị trường)
  - ⏰ "Tin Nhanh Crypto: (HH:MM)" + nhiều mục "🔥 ..." + "Xem thêm 👉👉👉" (03/2024+) -> tách từng mục
  - 🟠 tiêu đề bài viết crypto + "👉👉👉" (07/2025+)
  - 🚨 tin nóng: tiêu đề + nội dung (03/2026+)
  - 🔶 "Kinh doanh:" tin doanh nghiệp Việt Nam
  - không ký hiệu: bài của chủ kênh (nhận định, quảng cáo)

Các bước: tách mục -> chuẩn hoá chữ (NFC, bỏ ký tự ẩn, nối số bị xuống dòng, bỏ "Xem thêm", emoji trang trí)
-> tách số liệu kinh tế "Thực tế / Trước đó / Kì vọng" -> gắn chủ đề (TOPICS) -> loại trùng
(cùng 80 ký tự đầu trong 6 giờ) -> đánh dấu tầng 0 (giữ / loại + lý do).

Dùng:
    python news_clean.py                           # đọc historical_data/telegram/coin369channel.jsonl
    import news_clean as nc; items = nc.load_items()
Kết quả: artifacts/news/items_clean.parquet, artifacts/news/clean_report.txt
"""

from __future__ import annotations

import json
import warnings
warnings.filterwarnings("ignore", message="This pattern is interpreted")
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
for p in (_ROOT, _ROOT / "model_training"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SRC = _ROOT / "historical_data" / "telegram" / "coin369channel.jsonl"
OUT = _ROOT / "artifacts" / "news"
ITEMS = OUT / "items_clean.parquet"

DEV_END = pd.Timestamp("2025-01-01", tz="UTC")
# chủ đề (regex, so khớp trên chữ thường trừ mã cổ phiếu TICKER)
TOPICS = {
    "crypto": r"bitcoin|\bBTC\b|crypto|tiền (điện tử|mã hóa|mã hoá|số|ảo)|Binance|Coinbase|Ethereum|\bETH\b|stablecoin|Tether|USDT|"
              r"blockchain|altcoin|\bETF bitcoin|ETF giao ngay|Grayscale|BlackRock.*ETF|MicroStrategy|Saylor|\bCZ\b|Changpeng|"
              r"Ripple|\bXRP\b|Solana|FTX|Bybit|OKX",
    "fed": r"\bFed\b|Cục Dự trữ Liên bang|Powell|FOMC|Waller|Williams|Bowman|Jefferson|Kashkari|Bostic|Daly|Goolsbee|"
           r"Barkin|Logan|Mester|Harker|Hammack|Musalem|Schmid|Cook\b|Barr\b|lãi suất (của )?Mỹ|biên bản cuộc họp",
    "us_data": r"CPI|PCE|phi nông nghiệp|bảng lương|tỷ lệ thất nghiệp|trợ cấp thất nghiệp|JOLTS|ADP|GDP.*Mỹ|Mỹ.*GDP|"
               r"doanh số bán lẻ|PMI.*Mỹ|ISM|PPI|niềm tin người tiêu dùng|Michigan|lạm phát.*Mỹ|Mỹ.*lạm phát",
    "us_rates_usd": r"lợi suất.*(kho bạc|trái phiếu.*Mỹ)|trái phiếu kho bạc|chỉ số đô la|\bDXY\b|đồng đô la|đồng USD",
    "us_politics": r"Trump|Biden|Harris|Nhà Trắng|thuế quan|Quốc hội Mỹ|Hạ viện|Thượng viện|trần nợ|đóng cửa chính phủ|"
                   r"Yellen|Bessent|Bộ Tài chính Mỹ",
    "geo": r"Israel|Ukraine|\bNga\b|Putin|Zelensky|Iran|Hamas|Gaza|Hezbollah|Houthi|Biển Đỏ|Trung Đông|tên lửa|chiến tranh|"
           r"quân đội|không kích|Triều Tiên|Đài Loan",
    "other_cb": r"\bECB\b|Ngân hàng Trung ương Châu Âu|Lagarde|\bBOJ\b|Ngân hàng Nhật Bản|Ueda|\bBOE\b|Ngân hàng Anh|Bailey|"
                r"PBOC|Ngân hàng Nhân dân Trung Quốc|SNB|RBA|Ngân hàng Canada",
    "china": r"Trung Quốc|Bắc Kinh|Tập Cận Bình|\bNDT\b|nhân dân tệ|Hồng Kông|Hang Seng",
    "commodities": r"\bvàng\b|giá dầu|dầu thô|\bWTI\b|Brent|OPEC|khí đốt|EIA",
    "equities": r"cổ phiếu|chứng khoán|S&P|Nasdaq|Dow Jones|\bIPO\b|lợi nhuận quý|doanh thu|\([A-Z]{1,5}(\.[A-Z])?\)|Nvidia|Tesla|Apple",
    "price_report": r"(tăng|giảm|lên|xuống)\s+(hơn\s+)?[\d\.,]+\s*%|mức (cao|thấp) nhất (kể từ|trong)|giao ngay (tăng|giảm)|"
                    r"điểm cơ bản|bị thanh lý|thanh lý",
    "preview": r"xem trước|lịch (sự kiện|kinh tế)|sự kiện (lớn|quan trọng).*(tuần|ngày)|dữ liệu quan trọng",
    "promo": r"youtu\.?be|đăng ký|Rich Kid|RKT|học viên|khóa học|khoá học|inbox|liên hệ|airdrop|giveaway|t\.me/",
}

FMT_RULES = [("⏰", "crypto_digest"), ("🟠", "crypto_headline"), ("🚨", "breaking"), ("🔴", "do"), ("🔹", "xanh_nho"),
             ("🔵", "xanh"), ("🔶", "kinh_doanh_vn")]
ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
DECOR = re.compile(r"[👉🔥⚡️💥🆘✅❗️‼️📌📢🔔⭐️🌟🎉👇👆➡️]+")
NUM_BREAK = re.compile(r"(\d)[\.,]\s*\n\s*(\d{3})")
TIME_HM = re.compile(r"^\s*\*?\s*(\d{1,2}):(\d{2})\s*:?\s*")
SOURCE_TAIL = re.compile(r"\(\s*(Bloomberg|Reuters|CNBC|WSJ|Fox[^)]*|Báo chí liên quan|Kênh[^)]*|Tân Hoa Xã|CCTV|Nikkei|FT|AP|AFP)\s*\)\s*$", re.I)
TICKER = r"\([A-Z]{1,5}(\.[A-Z])?\)"
ECON = (r"(?i)dự kiến (là|ở mức)?\s*-?[\d\.,]+|giá trị trước đó|\bPMI\b|\bCPI\b|\bPPI\b|lạm phát|\bGDP\b|thất nghiệp|"
        r"bảng lương|doanh số bán lẻ|thu nhập bình quân|lãi suất (chuẩn|cơ bản|chính sách)|ngân hàng trung ương")
PROMO = re.compile(r"youtu\.?be|đăng ký|Rich Kid|RKT|học viên|khóa học|khoá học|inbox|liên hệ|airdrop|giveaway|t\.me/|zalo", re.I)


DATA_RE = re.compile(r"^(?P<name>.*?)\s*(?:Thực tế|Act)\s*:\s*(?P<act>[-+]?[\d\.,]+|-)?\s*%?\s*"
                     r"(?:(?:Trước đó|Pre)\s*:\s*(?P<prev>[-+]?[\d\.,]+|-)?\s*%?\s*)?"
                     r"(?:(?:K[ìi] vọng|Exp)\s*:\s*(?P<exp>[-+]?[\d\.,]+|-)?)?", re.S)


# dạng văn xuôi: "CPI cơ bản ở Hoa Kỳ trong tháng 3 là 2,6%, dự kiến 2,7%, giá trị trước đó 2,5%"
PROSE_RE = re.compile(r"^(?P<name>.{8,140}?)\s+(?:là|ở mức|đạt|ghi nhận)\s+(?P<act>[-+]?\d[\d\.,]*)\s*(?:%|nghìn|triệu|tỷ|điểm)?"
                      r"[^.;]{0,40}?,?\s*(?:so với\s+)?(?:mức\s+)?(?:dự kiến|kỳ vọng|dự báo)(?: là| ở mức)?\s+(?P<exp>[-+]?\d[\d\.,]*)"
                      r"\s*(?:%|nghìn|triệu|tỷ)?[^.;]{0,30}?(?:giá trị trước đó|trước đó|kỳ trước)(?: là| được điều chỉnh[^\d-]*)?"
                      r"\s*(?P<prev>[-+]?\d[\d\.,]*)")
COUNTRIES = {"Mỹ": "US", "Hoa Kỳ": "US", "Eurozone": "EU", "Khu vực đồng Euro": "EU", "khu vực đồng euro": "EU", "Đức": "DE",
             "Nhật Bản": "JP", "Canada": "CA", "Vương quốc Anh": "UK", "Anh": "UK", "Pháp": "FR", "Úc": "AU", "New Zealand": "NZ",
             "Trung Quốc": "CN", "Hàn Quốc": "KR", "Thụy Sĩ": "CH", "Ấn Độ": "IN", "Ý": "IT", "Tây Ban Nha": "ES", "Nga": "RU"}
COUNTRY_RE = re.compile(r"(?<!\w)(" + "|".join(sorted(map(re.escape, COUNTRIES), key=len, reverse=True)) + r")(?!\w)")
US_KEY = {"cpi_core": r"CPI (cốt lõi|cơ bản)|CPI lõi", "cpi": r"\bCPI\b", "pce": r"\bPCE\b", "nfp": r"phi nông nghiệp",
          "unemployment": r"tỷ lệ thất nghiệp", "claims": r"trợ cấp thất nghiệp", "gdp": r"\bGDP\b", "retail": r"bán lẻ",
          "ism": r"\bISM\b", "ppi": r"\bPPI\b", "jolts": r"JOLTS|việc làm trống", "adp": r"\bADP\b",
          "michigan": r"Michigan", "fed_rate": r"lãi suất.*(Fed|Cục Dự trữ)|(Fed|FOMC).*lãi suất"}


def num(s) -> float:
    """'3.70' / '-358,7' / '1,234' / '1.234,5' -> float"""
    if not isinstance(s, str) or s.strip(" -+") == "":
        return np.nan
    s = s.strip().rstrip(".,")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", "") if re.fullmatch(r"[-+]?[1-9]\d{0,2}(,\d{3})+", s) else s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").translate(ZW)
    s = NUM_BREAK.sub(r"\1.\2", s)
    s = re.sub(r"Xem thêm\s*[👉\s]*$", "", s.strip())
    return s


def flat(s: str) -> str:
    return re.sub(r"\s+", " ", DECOR.sub(" ", s)).strip(" -–:/|")


def detect_format(text: str) -> str:
    t = text.lstrip()
    for k, v in FMT_RULES:
        if t.startswith(k):
            return v
    if t.startswith("Tin nhanh"):
        return "tin_nhanh"
    return "chu_kenh"


EMBED = re.compile(r"\s*(?=[🔴🔹🔵🔶]\ufe0f?\s*(?:Kinh doanh\s*:\s*)?\d{1,2}:\d{2}\s*:)")


def split_items(text: str, fmt: str) -> list[tuple[str, float, str]]:
    """-> [(nội dung, phút trong ngày ghi trong tin hoặc NaN, định dạng của mục)]
    Từ 2026 kênh gộp nhiều mục "🔴 HH:MM: ..." vào 1 tin -> tách theo ký hiệu nhúng, mỗi mục giữ ký hiệu riêng."""
    if fmt in ("do", "xanh_nho", "xanh", "kinh_doanh_vn", "chu_kenh"):
        parts = [p for p in EMBED.split(norm_text(text)) if p.strip()]
        if len(parts) > 1:
            out = []
            for p in parts:
                f = detect_format(p)
                out += split_items_one(p, f if f != "chu_kenh" else fmt)
            return out
    return split_items_one(text, fmt)


def split_items_one(text: str, fmt: str) -> list[tuple[str, float, str]]:
    t = norm_text(text)
    if fmt == "tin_nhanh":
        body = re.sub(r"^\s*Tin nhanh\s*:?", "", t)
        parts = re.split(r"\n\s*\*\s*(?=\d{1,2}:\d{2})", "\n" + body)
        out = []
        for p in parts:
            p = p.strip(" *\n")
            if not p:
                continue
            m = TIME_HM.match(p)
            out.append((p[m.end():] if m else p, int(m.group(1)) * 60 + int(m.group(2)) if m else np.nan, fmt))
        return out
    if fmt == "crypto_digest":
        m = re.search(r"\((\d{1,2}):(\d{2})\)", t)
        tm = int(m.group(1)) * 60 + int(m.group(2)) if m else np.nan
        body = re.sub(r"^.*?Tin Nhanh Crypto\s*:?\s*(\(\d{1,2}:\d{2}\))?", "", t, count=1, flags=re.S)
        parts = [p.strip() for p in re.split(r"🔥", body)]
        return [(p, tm, fmt) for p in parts if p and not re.fullmatch(r"[\s👉]*", p)]
    # các dạng 1 mục
    body = t[1:].lstrip("️ ") if fmt in ("crypto_headline", "breaking", "do", "xanh_nho", "xanh", "kinh_doanh_vn") else t
    if fmt == "kinh_doanh_vn":
        body = re.sub(r"^\s*Kinh doanh\s*:?", "", body)
    m = TIME_HM.match(body)
    tm = int(m.group(1)) * 60 + int(m.group(2)) if m else np.nan
    if m:
        body = body[m.end():]
    return [(body, tm, fmt)]


def build_items(src: Path = SRC, log=print) -> pd.DataFrame:
    t0 = time.time()
    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = r.get("text") or ""
            fmt = detect_format(text)
            items = split_items(text, fmt)
            for k, (body, tm, ifmt) in enumerate(items):
                b = flat(body)
                src_m = SOURCE_TAIL.search(b)
                rows.append((int(r["msg_id"]), k, len(items), r["date"], r.get("edit_date"), r.get("views"), ifmt,
                             tm, SOURCE_TAIL.sub("", b).strip() if src_m else b, src_m.group(1) if src_m else None,
                             r.get("collected_at")))
    log(f"đọc + tách: {len(rows):,} mục từ file ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows, columns=["msg_id", "item", "n_items", "date", "edit_date", "views", "fmt", "time_in_text_min",
                                     "body", "source", "collected_at"])
    df = df.sort_values("collected_at").drop_duplicates(["msg_id", "item"], keep="last").drop(columns="collected_at")
    df["date"] = pd.to_datetime(df.date, utc=True)
    df["edit_date"] = pd.to_datetime(df.edit_date, utc=True, errors="coerce")
    df = df.sort_values(["date", "msg_id", "item"]).reset_index(drop=True)
    df["edit_after_min"] = (df.edit_date - df.date).dt.total_seconds() / 60
    df["edited_late"] = df.edit_after_min > 10               # sửa sau > 10 phút: có thể thêm thông tin về sau
    loc = df.date.dt.tz_convert("Asia/Ho_Chi_Minh")
    post_min = loc.dt.hour * 60 + loc.dt.minute
    df["delay_min"] = ((post_min - df.time_in_text_min) % 1440).where(df.time_in_text_min.notna())
    # số liệu kinh tế
    d = df.body.str.extract(DATA_RE)
    df["data_name"] = d["name"].str.strip(" :-")
    for c in ("act", "prev", "exp"):
        df[f"data_{c}"] = d[c].map(num)
    df["data_src"] = np.where(df.data_name.notna(), "table", "")
    rest = df.data_name.isna()
    pr = df.loc[rest, "body"].str.extract(PROSE_RE)
    pr = pr[pr.act.notna()]
    df.loc[pr.index, "data_name"] = pr["name"].str.strip(" :-,")
    for c in ("act", "prev", "exp"):
        df.loc[pr.index, f"data_{c}"] = pr[c].map(num)
    df.loc[pr.index, "data_src"] = "prose"
    df["is_data"] = df.data_name.notna()
    df["is_preview"] = ~df.is_data & df.body.str.contains(r"K[ìi] vọng\s*:|Exp\s*:", regex=True, na=False)  # lịch trước giờ công bố
    log(f"chuẩn hoá + số liệu: {time.time() - t0:.0f}s")
    # chủ đề
    low = df.body.str.lower()                       # re.IGNORECASE với unicode rất chậm -> so khớp trên chữ thường
    for k, pat in TOPICS.items():
        cs = TICKER if TICKER in pat else None
        pl = pat.replace("|" + TICKER, "").replace(TICKER, "").lower() if cs else pat.lower()
        hit = low.str.contains(pl, regex=True, na=False)
        if cs:
            hit |= df.body.str.contains(cs, regex=True, na=False)
        df[f"t_{k}"] = hit
    df.loc[df.fmt.isin(["crypto_digest", "crypto_headline"]), "t_crypto"] = True
    df["t_promo"] = df.t_promo | (df.fmt.eq("chu_kenh") & df.body.str.contains(PROMO, na=False))
    tcols = [f"t_{k}" for k in TOPICS]
    df["n_topics"] = df[tcols].sum(axis=1)
    df["t_other"] = df.n_topics.eq(0)
    log(f"chủ đề: {time.time() - t0:.0f}s")
    # trùng: cùng 80 ký tự đầu (chuẩn hoá) trong 6 giờ
    key = df.body.str.lower().str.replace(r"[^\w]+", " ", regex=True).str.strip().str[:80].to_numpy()
    tns = df.date.astype("int64").to_numpy()
    last: dict[str, int] = {}
    dup = np.zeros(len(df), bool)
    six = 6 * 3600 * 10**9
    for i in range(len(df)):
        k = key[i]
        if not k:
            dup[i] = True
            continue
        p = last.get(k)
        if p is not None and tns[i] - p <= six:
            dup[i] = True
        last[k] = tns[i]
    df["is_dup"] = dup
    # tầng 0
    econ = df.body.str.contains(ECON, regex=True, na=False)
    macro = econ | df[["t_fed", "t_us_data", "t_us_rates_usd", "t_us_politics", "t_other_cb", "t_geo", "t_china"]].any(axis=1)
    crypto_policy = df.t_crypto & df.body.str.contains(r"SEC|ETF|quy định|luật|dự luật|phê duyệt|cấm|hack|tấn công|"
                                                       r"vụ kiện|kiện|Trump|Fed|dự trữ|mua vào|bán ra|niêm yết|hủy niêm yết",
                                                       case=False, regex=True, na=False)
    reason = np.select(
        [df.is_dup, df.body.str.len() < 20, df.t_promo, df.fmt.eq("kinh_doanh_vn"),
         df.t_price_report & ~macro & ~df.is_data & ~crypto_policy,
         df.t_equities & ~macro & ~df.t_crypto & ~df.is_data],
        ["trùng", "quá ngắn", "quảng cáo / chủ kênh", "doanh nghiệp VN", "tường thuật giá", "cổ phiếu riêng lẻ"], default="")
    df["drop_reason"] = reason
    df["keep_stage0"] = df.drop_reason.eq("")
    log(f"trùng + tầng 0: {time.time() - t0:.0f}s")
    return df


def econ_events(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng số liệu kinh tế (1 dòng = 1 mục có Thực tế / Dự kiến / Trước đó). Cùng 1 lần công bố có thể xuất hiện
    2 lần (bảng + văn xuôi) -> `release_id` gom các mục cùng quốc gia + cùng nhóm chỉ số trong 10 phút."""
    e = df[df.is_data & ~df.is_dup].copy()
    e["name"] = e.data_name.str.replace(r"^.*(?:dự kiến|trước đó)[^.]*\.\s*", "", regex=True).str.strip(" .:")
    e["country"] = e.name.str.extract(COUNTRY_RE)[0].map(COUNTRIES)
    e.loc[e.country.isna() & e.name.str.contains(r"\bFed\b|ISM|ADP|Michigan|API|EIA|Conference", regex=True), "country"] = "US"
    e["us_key"] = ""
    for k, rx in reversed(list(US_KEY.items())):      # khoá đầu danh sách ưu tiên (cpi_core trước cpi)
        hit = e.country.eq("US") & e.name.str.contains(rx, case=False, regex=True)
        e.loc[hit, "us_key"] = k
    e["surprise"] = e.data_act - e.data_exp
    e["key"] = np.where(e.us_key.ne(""), "US:" + e.us_key, e.country.fillna("?") + ":" + e.name.str.lower().str[:25])
    e = e.sort_values("date")
    gap = e.groupby("key").date.diff().dt.total_seconds().gt(600) | e.groupby("key").date.diff().isna()
    e["release_id"] = gap.cumsum()
    cols = ["date", "msg_id", "item", "fmt", "data_src", "country", "us_key", "name", "data_act", "data_exp", "data_prev",
            "surprise", "release_id", "edited_late", "body"]
    return e[cols].reset_index(drop=True)


def report(df: pd.DataFrame) -> str:
    L = []
    L.append(f"Mục tin: {len(df):,} (từ {df.msg_id.nunique():,} tin), {df.date.min():%Y-%m-%d} -> {df.date.max():%Y-%m-%d}")
    L.append("Định dạng: " + ", ".join(f"{k} {v:,}" for k, v in df.fmt.value_counts().items()))
    L.append(f"Sửa sau > 10 phút: {df.edited_late.mean():.1%} | số liệu kinh tế: {int(df.is_data.sum()):,} "
             f"(bảng {int(df.data_src.eq('table').sum()):,}, văn xuôi {int(df.data_src.eq('prose').sum()):,})")
    L.append("Lý do loại (tầng 0): " + ", ".join(f"{k or 'GIỮ'} {v:,}" for k, v in df.drop_reason.value_counts().items()))
    K = df[df.keep_stage0]
    L.append(f"Giữ sau tầng 0: {len(K):,} ({len(K) / len(df):.1%}); trước 2025: {(K.date < DEV_END).sum():,}, "
             f"từ 2025: {(K.date >= DEV_END).sum():,}")
    L.append("Chủ đề trong phần giữ: " + ", ".join(f"{k} {int(K[f't_{k}'].sum()):,}" for k in list(TOPICS) + ["other"]))
    return "\n".join(L)


def save(df: pd.DataFrame, path: Path = ITEMS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)                                   # ghi xong mới thay file cũ
    return path


def load_items(path: Path = ITEMS) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["body"] = df.body.fillna("")
    return df


if __name__ == "__main__":
    items = build_items()
    rep = report(items)
    ev = econ_events(items)
    rep += (f"\nSố liệu kinh tế (không trùng): {len(ev):,} mục, {ev.release_id.nunique():,} lần công bố; Mỹ "
            f"{int(ev.country.eq('US').sum()):,}; chỉ số Mỹ quan trọng: "
            + ", ".join(f"{k} {v}" for k, v in ev[ev.us_key.ne('')].groupby('us_key').release_id.nunique().items()))
    print(rep)
    (OUT / "clean_report.txt").write_text(rep, encoding="utf-8")
    ev.to_csv(OUT / "econ_events.csv", index=False, encoding="utf-8-sig")
    print("lưu:", save(items))
