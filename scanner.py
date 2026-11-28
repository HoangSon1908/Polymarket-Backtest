import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
import streamlit as st
from urllib3.util.retry import Retry

st.set_page_config(
    page_title="Polymarket Backtest Extractor",
    page_icon="🧬",
    layout="centered",
)

BASE_V2_URL = "https://data-api.polymarket.com/v2"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
POSITIONS_URL = f"{BASE_V2_URL}/positions"
ACTIVITY_URL = f"{BASE_V2_URL}/activity"
DEFAULT_WALLET = "0x474D6503CD4867D611b012ED6c8a6fAAaf5e6b5f"
VN_TZ = timezone(timedelta(hours=7))
EVM_ADDR_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
_THREAD_LOCAL = threading.local()


def get_http_session() -> requests.Session:
    """Mỗi thread tái sử dụng 1 HTTP Keep-Alive Session kèm Retry tự động khi gặp 429/5xx."""
    sess = getattr(_THREAD_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        retry_cfg = Retry(
            total=4,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=retry_cfg)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        sess.headers.update({"Accept": "application/json", "User-Agent": "Polymarket-Backtest-Extractor/2.1"})
        _THREAD_LOCAL.session = sess
    return sess


def sanitize_filename_part(name: str) -> str:
    """Làm sạch chuỗi để dùng làm tên file CSV an toàn trên mọi hệ điều hành."""
    cleaned = re.sub(r'[\\/*?:"<>|\s]+', "_", str(name or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.")
    return cleaned


def _is_valid_profile_name(val: str) -> bool:
    s = str(val or "").strip()
    if not s:
        return False
    if s.lower().startswith("0x") and (len(s) >= 10 or "..." in s):
        return False
    return True


def lookup_profile_name_by_wallet(wallet: str) -> str:
    """Tra cứu nhanh username/profile name trên Polymarket (chỉ gọi khi payload chưa có tên)."""
    w = wallet.strip()
    w_lower = w.lower()
    sess = get_http_session()

    # Lớp 1: Gamma /public-profile
    try:
        resp = sess.get(f"{GAMMA_API_BASE}/public-profile", params={"address": w_lower}, timeout=5)
        if resp.ok and isinstance(resp.json(), dict):
            data = resp.json()
            for key in ("name", "pseudonym", "userName", "user_name", "username"):
                cand = str(data.get(key) or "").strip()
                if _is_valid_profile_name(cand):
                    return cand
    except Exception:
        pass

    # Lớp 2: Data API v2 /leaderboard
    try:
        resp = sess.get(
            f"{BASE_V2_URL}/leaderboard",
            params={"category": "OVERALL", "timePeriod": "ALL", "user": w_lower, "limit": 5},
            timeout=5,
        )
        if resp.ok:
            payload = resp.json()
            rows = payload.get("data") or [] if isinstance(payload, dict) else (payload or [])
            for item in rows:
                pw = str(item.get("proxy_wallet") or item.get("proxyWallet") or item.get("user") or "").strip().lower()
                if pw == w_lower:
                    for key in ("user_name", "userName", "name", "pseudonym"):
                        cand = str(item.get(key) or "").strip()
                        if _is_valid_profile_name(cand):
                            return cand
    except Exception:
        pass

    # Lớp 3: Gamma /public-search
    try:
        resp = sess.get(
            f"{GAMMA_API_BASE}/public-search",
            params={"q": w_lower, "search_profiles": "true", "limit_per_type": 5},
            timeout=5,
        )
        if resp.ok:
            profiles = (resp.json() or {}).get("profiles") or []
            for p in profiles:
                pw = str(p.get("proxyWallet") or p.get("proxy_wallet") or p.get("address") or "").strip().lower()
                if pw == w_lower:
                    for key in ("name", "user_name", "userName", "pseudonym"):
                        cand = str(p.get(key) or "").strip()
                        if _is_valid_profile_name(cand):
                            return cand
    except Exception:
        pass

    return ""


def resolve_wallet_only(raw_input: str) -> tuple[str, str, str]:
    """
    Phân giải địa chỉ ví 0x... tức thì (không tốn HTTP call nếu đã nhập 0x...).
    Trả về: (wallet_address, preset_profile_name, error_message)
    """
    cleaned = raw_input.strip()
    if not cleaned:
        return "", "", "Vui lòng nhập địa chỉ ví (0x...) hoặc @username."

    cache: dict[str, tuple[str, str]] = st.session_state.setdefault("wallet_profile_cache", {})
    cache_key = cleaned.lower()
    if cache_key in cache:
        w_cached, n_cached = cache[cache_key]
        return w_cached, n_cached, ""

    if EVM_ADDR_RE.match(cleaned):
        return cleaned, "", ""

    handle = cleaned.lstrip("@").strip()
    resolved_wallet = ""
    found_name = handle
    sess = get_http_session()

    try:
        resp = sess.get(
            f"{GAMMA_API_BASE}/public-search",
            params={"q": handle, "search_profiles": "true", "limit_per_type": 10},
            timeout=8,
        )
        if resp.ok:
            profiles = (resp.json() or {}).get("profiles") or []
            for p in profiles:
                pw = str(p.get("proxyWallet") or p.get("proxy_wallet") or p.get("address") or "").strip()
                uname = str(p.get("name") or p.get("pseudonym") or "").strip()
                if EVM_ADDR_RE.match(pw) and uname.lower() == handle.lower():
                    resolved_wallet = pw
                    found_name = uname or handle
                    break
            if not resolved_wallet:
                for p in profiles:
                    pw = str(p.get("proxyWallet") or p.get("proxy_wallet") or p.get("address") or "").strip()
                    if EVM_ADDR_RE.match(pw):
                        resolved_wallet = pw
                        cand = str(p.get("name") or p.get("pseudonym") or "").strip()
                        found_name = cand if _is_valid_profile_name(cand) else handle
                        break
    except Exception:
        pass

    if not resolved_wallet:
        try:
            page = sess.get(f"https://polymarket.com/@{handle}", timeout=8)
            if page.ok:
                m_json = re.search(r'"proxyWallet"\s*:\s*"(0x[a-fA-F0-9]{40})"', page.text)
                if m_json:
                    resolved_wallet = m_json.group(1)
        except Exception:
            pass

    if not resolved_wallet:
        return "", "", f"Không tìm thấy ví Polymarket cho '{cleaned}'. Hãy dán trực tiếp địa chỉ 0x..."

    auto_name = sanitize_filename_part(found_name) or f"{resolved_wallet[:6]}_{resolved_wallet[-4:]}"
    cache[cache_key] = (resolved_wallet, auto_name)
    return resolved_wallet, auto_name, ""


def fetch_v2_cursor(endpoint: str, base_params: dict, max_pages: int = 150) -> list[dict]:
    """Kéo toàn bộ dữ liệu theo chuẩn Polymarket Data API v2 (Cursor-only + chống lặp cursor)."""
    rows = []
    params = dict(base_params)
    params["limit"] = min(int(params.get("limit", 500)), 500)
    anchor_keys = [k for k in ("user", "status", "type", "side") if k in params]
    seen_cursors: set[str] = set()
    sess = get_http_session()

    for _ in range(max_pages):
        res = sess.get(endpoint, params=params, timeout=20)
        if res.status_code == 429:
            time.sleep(1.5)
            continue
        res.raise_for_status()
        payload = res.json()

        batch = payload.get("data") if isinstance(payload, dict) else payload
        if batch:
            rows.extend(batch)

        if not isinstance(payload, dict):
            break

        pagination = payload.get("pagination") or {}
        next_cursor = pagination.get("next_cursor")
        has_more = pagination.get("has_more", False)

        if not next_cursor or not has_more or next_cursor in seen_cursors:
            break

        seen_cursors.add(next_cursor)
        params = {k: base_params[k] for k in anchor_keys}
        params["cursor"] = next_cursor

    return rows


def to_epoch(val) -> float:
    if pd.isna(val) or val is None or val == "":
        return np.nan
    try:
        ts = float(val)
        if ts > 1e11:
            ts /= 1000.0
        return ts if ts > 0 else np.nan
    except Exception:
        try:
            return pd.to_datetime(val, utc=True).timestamp()
        except Exception:
            return np.nan


def parse_market_type(title: str) -> str:
    t = title.lower()
    if "temperature" in t:
        return "WEATHER_TEMP"
    if "map handicap" in t:
        return "MAP_HANDICAP"
    if "rounds handicap" in t:
        return "ROUNDS_HANDICAP"
    if re.search(r"map\s*[123]\s*winner", t):
        return "SINGLE_MAP"
    if "games total" in t or "total maps" in t or "o/u" in t:
        return "TOTAL_MAPS_OU"
    if "(bo1)" in t:
        return "MATCH_BO1"
    if "(bo5)" in t:
        return "MATCH_BO5"
    if "(bo3)" in t:
        return "MATCH_BO3"
    return "OTHER"


def parse_tournament(title: str) -> str:
    t = title.strip()
    if "temperature in " in t.lower():
        m = re.search(r"temperature in ([A-Za-z\s\.]+?)\s+be\b", t, flags=re.IGNORECASE)
        if m:
            return f"Weather:{m.group(1).strip()}"
    if " - " in t:
        suffix = t.split(" - ", 1)[1].strip()
        for stage in (
            " Group", " Playoffs", " Play-In", " Closed Qualifier",
            " Open Qualifier", " Swiss", " Stage", " Series #", " Cup #",
        ):
            if stage.lower() in suffix.lower():
                idx = suffix.lower().index(stage.lower())
                suffix = suffix[:idx].strip()
        return suffix if suffix else "Unknown"
    if any(k in t.lower() for k in ("handicap", "map 1", "map 2", "map 3", "games total")):
        return "Side_Market"
    return "Unknown"


def _collapse_split_merge_group(g: pd.DataFrame) -> pd.Series:
    """Gộp 2 chân (Yes + No) của cùng 1 market SETTLED_SPLIT_MERGE thành 1 vị thế theo condition_id."""
    g_sorted = g.sort_values("true_cost", ascending=False)
    first = g_sorted.iloc[0]

    tot_cost = round(float(g["true_cost"].sum()), 2)
    tot_realized = round(float(g["realized_pnl"].sum()), 2)
    tot_unrealized = round(float(g["unrealized_pnl"].sum()), 2)
    tot_pnl = round(float(g["total_pnl"].sum()), 2)
    roi = round((tot_pnl / tot_cost) * 100.0, 2) if tot_cost > 0.05 else np.nan

    b_cnt = int(g["buy_count"].sum())
    s_cnt = int(g["sell_count"].sum())
    sp_cnt = int(g["split_count"].max())
    mg_cnt = int(g["merge_count"].max())

    outcomes_held = "+".join(sorted(o for o in g["outcome"].astype(str).unique() if o))
    tokens_joined = "|".join(sorted(t for t in g["token_id"].astype(str).unique() if t))

    if mg_cnt > 0:
        exit_st = "MERGED_2_SIDES"
    elif s_cnt > 0:
        exit_st = "SPLIT_SOLD_SIDE"
    else:
        exit_st = "SPLIT_HOLD"

    w_price = float((g["avg_price"] * g["true_cost"]).sum() / tot_cost) if tot_cost > 0 else 0.5
    valid_entries = [x for x in g["entry_time_vn"].astype(str).tolist() if x]
    valid_lasts = [x for x in g["last_time_vn"].astype(str).tolist() if x]

    row = first.copy()
    row["first_ts"] = g["first_ts"].min()
    row["last_ts"] = g["last_ts"].max()
    row["entry_time_vn"] = min(valid_entries) if valid_entries else ""
    row["last_time_vn"] = max(valid_lasts) if valid_lasts else ""
    row["token_id"] = tokens_joined
    row["outcome"] = f"BOTH ({outcomes_held})" if outcomes_held else "BOTH"
    row["is_hedged"] = True
    row["avg_price"] = round(w_price, 4)
    row["true_cost"] = tot_cost
    row["realized_pnl"] = tot_realized
    row["unrealized_pnl"] = tot_unrealized
    row["total_pnl"] = tot_pnl
    row["roi_pct"] = roi
    row["is_win"] = bool(tot_pnl > 0)
    row["buy_count"] = b_cnt
    row["sell_count"] = s_cnt
    row["split_count"] = sp_cnt
    row["merge_count"] = mg_cnt
    row["avg_clip_usdc"] = round(tot_cost / max(b_cnt + sp_cnt, 1), 2)
    row["accum_hours"] = round(float(g["accum_hours"].max()), 2)
    row["hold_hours"] = round(float(g["hold_hours"].max()), 2)
    row["exit_style"] = exit_st
    return row


def build_ai_slim_csv(
    df: pd.DataFrame,
    clean_only: bool = True,
    recent_months: int = 0,
    min_whale_cost: float = 0.0,
) -> pd.DataFrame:
    """
    Nén tối đa file CSV cho Gemini (~10-13 tokens/dòng, giảm ~60% so với bản cũ):
    - Cắt bỏ giây (:SS), xóa month & last_time_vn trùng lặp.
    - Thay chuỗi hex condition_id bằng mã nhóm ngắn H1, H2... chỉ trên kèo có Hedge 2 đầu.
    - Xóa hậu tố giải đấu đã tách ở cột tournament & tiền tố boilerplate trong title.
    - Xóa cột tính toán suy diễn (roi_pct, avg_clip_usdc) nhưng giữ đủ 100% biến gốc để chạy mọi phép kiểm định.
    """
    sub = df.copy()
    if clean_only and "is_backtest_clean" in sub.columns:
        sub = sub[sub["is_backtest_clean"]].copy()
    if recent_months > 0 and "month" in sub.columns:
        valid_months = sorted([m for m in sub["month"].unique() if m != "Unknown"])
        if len(valid_months) > recent_months:
            sub = sub[sub["month"].isin(set(valid_months[-recent_months:]))].copy()
    if min_whale_cost > 0 and "true_cost" in sub.columns:
        sub = sub[sub["true_cost"] >= min_whale_cost].copy()

    if sub.empty:
        return pd.DataFrame()

    sub["entry_time_vn"] = sub["entry_time_vn"].astype(str).str.slice(0, 16)

    hedged_mask = sub["is_hedged"].fillna(False).astype(bool)
    hedged_cids = sub.loc[hedged_mask, "condition_id"].unique()
    cid_map = {cid: f"H{i+1}" for i, cid in enumerate(hedged_cids)}
    sub["hedge_id"] = np.where(hedged_mask, sub["condition_id"].map(cid_map).fillna(""), "")

    def _compact_title(row) -> str:
        t = str(row["title"] or "")
        if row.get("tournament") not in ("", "Unknown", "Side_Market") and " - " in t:
            t = t.split(" - ", 1)[0].strip()
        return re.sub(
            r"^(Counter-Strike:\s*|Dota 2:\s*|League of Legends:\s*|Valorant:\s*|Will the highest temperature in\s*)",
            "",
            t,
            flags=re.IGNORECASE,
        )

    sub["title"] = sub.apply(_compact_title, axis=1)
    sub["is_win"] = np.where(sub["is_win"].fillna(False), "1", "")
    sub["record_type"] = sub["record_type"].replace({
        "SETTLED_CLOB": "CLOB",
        "SETTLED_SPLIT_MERGE": "SM",
        "SETTLED_ESTIMATED": "EST",
    })
    sub["avg_price"] = sub["avg_price"].round(3)

    keep_cols = [
        "entry_time_vn", "record_type", "hedge_id", "title", "outcome",
        "market_type", "tournament", "avg_price", "true_cost", "total_pnl",
        "is_win", "buy_count", "accum_hours", "hold_hours",
    ]
    return sub[[c for c in keep_cols if c in sub.columns]].reset_index(drop=True)


def build_ai_prompt_digest(df: pd.DataFrame, wallet: str, label: str) -> str:
    """
    Tạo bản tóm tắt kiểm toán định lượng (~500-600 tokens) gồm:
    - Thống kê tổng hợp, Z-score trọng số vốn & PnL_exTop5 (chống lệ thuộc kèo độc đắc).
    - Kiểm định phân tách thời gian Out-of-Sample (70% đầu vs 30% gần nhất) chống overfit.
    - Phân phối theo Tháng, Dải giá, Quy mô vốn ví gốc, Market Type & Top Giải đấu.
    """
    clean = df[df["is_backtest_clean"]].copy() if "is_backtest_clean" in df.columns else df.copy()
    if clean.empty:
        return f"AUDIT SUMMARY | {label} ({wallet}): Không có dữ liệu kèo đã tất toán."

    def _calc_z(sub: pd.DataFrame) -> float:
        p = sub["avg_price"].clip(0.02, 0.98)
        c = sub["true_cost"].clip(lower=0.0)
        var_sum = float(((c ** 2) * (1.0 - p) / p).sum())
        return float(sub["total_pnl"].sum()) / np.sqrt(var_sum) if var_sum > 0 else 0.0

    def _agg_table(sub: pd.DataFrame, group_col: str, top_n: int = 8) -> str:
        g = (
            sub.groupby(group_col, dropna=False, observed=False)
            .agg(
                n=("total_pnl", "count"),
                wr=("is_win", "mean"),
                avg_p=("avg_price", "mean"),
                vol=("true_cost", "sum"),
                pnl=("total_pnl", "sum"),
            )
            .sort_values("vol", ascending=False)
            .head(top_n)
        )
        lines = [
            f"  - {idx}: N={int(r.n)}, WR={r.wr*100:.1f}%, AvgP={r.avg_p*100:.1f}c, "
            f"Vol=${r.vol:,.0f}, PnL=${r.pnl:+,.1f} (ROI {r.pnl/max(r.vol,1)*100:+.1f}%)"
            for idx, r in g.iterrows()
            if r.n > 0
        ]
        return "\n".join(lines) if lines else "  - N/A"

    clean["price_band"] = pd.cut(
        clean["avg_price"],
        bins=[-0.01, 0.35, 0.50, 0.60, 0.85, 1.01],
        labels=["<35c", "35-50c", "50-60c", "60-85c", ">85c"],
    )
    clean["size_band"] = pd.cut(
        clean["true_cost"],
        bins=[-1, 400, 2000, 10000, 20000, 1e9],
        labels=["<400", "400-2k", "2k-10k", "10k-20k", ">20k"],
    )

    tot_vol = float(clean["true_cost"].sum())
    tot_pnl = float(clean["total_pnl"].sum())
    z_all = _calc_z(clean)
    top5_pnl = float(clean.nlargest(5, "total_pnl")["total_pnl"].sum())
    ex_top5 = tot_pnl - top5_pnl
    med_accum = float(clean["accum_hours"].median())
    med_hold = float(clean["hold_hours"].median())
    hedge_share = float(clean.loc[clean["is_hedged"], "true_cost"].sum()) / max(tot_vol, 1.0) * 100.0

    # Phân tách thời gian Out-of-Sample (70% đầu vs 30% gần nhất)
    split_idx = int(len(clean) * 0.7)
    if split_idx >= 5 and (len(clean) - split_idx) >= 5:
        is_df = clean.iloc[:split_idx]
        oos_df = clean.iloc[split_idx:]
        oos_line = (
            f"[Kiểm định Thời gian 70/30 OOS]\n"
            f"  - In-Sample (70% đầu, N={len(is_df)}): WR={is_df['is_win'].mean()*100:.1f}%, "
            f"Vol=${is_df['true_cost'].sum():,.0f}, PnL=${is_df['total_pnl'].sum():+,.1f} "
            f"(ROI {is_df['total_pnl'].sum()/max(is_df['true_cost'].sum(),1)*100:+.1f}%, Z={_calc_z(is_df):.2f})\n"
            f"  - Out-of-Sample (30% gần nhất, N={len(oos_df)}): WR={oos_df['is_win'].mean()*100:.1f}%, "
            f"Vol=${oos_df['true_cost'].sum():,.0f}, PnL=${oos_df['total_pnl'].sum():+,.1f} "
            f"(ROI {oos_df['total_pnl'].sum()/max(oos_df['true_cost'].sum(),1)*100:+.1f}%, Z={_calc_z(oos_df):.2f})\n"
        )
    else:
        oos_line = ""

    return (
        f"AUDIT SUMMARY | {label} ({wallet})\n"
        f"Clean: N={len(clean)}, WR={clean['is_win'].mean()*100:.1f}%, AvgP={clean['avg_price'].mean()*100:.1f}c, "
        f"Vol=${tot_vol:,.0f}, PnL=${tot_pnl:+,.2f} (ROI {tot_pnl/max(tot_vol,1)*100:+.2f}%), Z={z_all:.2f}\n"
        f"Risk: PnL_exTop5=${ex_top5:+,.2f}, HedgeVol={hedge_share:.1f}%, MedAccum={med_accum:.1f}h, MedHold={med_hold:.1f}h\n"
        f"{oos_line}"
        f"[Theo Tháng]\n{_agg_table(clean, 'month', 6)}\n"
        f"[Theo Dải Giá]\n{_agg_table(clean, 'price_band', 5)}\n"
        f"[Theo Quy Mô Vốn Ví Gốc]\n{_agg_table(clean, 'size_band', 5)}\n"
        f"[Theo Market Type]\n{_agg_table(clean, 'market_type', 8)}\n"
        f"[Top Giải Đấu]\n{_agg_table(clean, 'tournament', 8)}"
    )


def build_backtest_ledger(wallet: str) -> tuple[pd.DataFrame, dict]:
    """Trích xuất sổ cái chuẩn theo Cách C từ /v2/positions & /v2/activity (song song đa luồng)."""
    w = wallet.strip()
    statuses = ("CLOSED", "REDEEMABLE", "REDEEMABLE_LOST", "MERGEABLE", "OPEN")

    def _pull_pos(st_val: str) -> list[dict]:
        batch = fetch_v2_cursor(POSITIONS_URL, {"user": w, "status": st_val, "limit": 500}, max_pages=120)
        for item in batch:
            item["_queried_status"] = st_val
        return batch

    # Chạy song song 5 luồng status + 1 luồng activity cùng lúc (~3x-4x nhanh hơn)
    with ThreadPoolExecutor(max_workers=6) as pool:
        pos_futures = [pool.submit(_pull_pos, st_val) for st_val in statuses]
        act_future = pool.submit(fetch_v2_cursor, ACTIVITY_URL, {"user": w, "limit": 500}, 150)

        raw_positions = []
        for fut in pos_futures:
            raw_positions.extend(fut.result())
        act_rows = act_future.result()

    if not raw_positions:
        return pd.DataFrame(), {
            "volume": 0.0, "pnl": 0.0, "markets": 0, "roi_pct": 0.0,
            "clean_volume": 0.0, "clean_pnl": 0.0, "clean_markets": 0,
            "clean_roi_pct": 0.0, "clean_winrate": 0.0, "clean_trades": 0,
            "detected_profile_name": "",
        }

    detected_profile_name = ""
    for item in raw_positions[:20]:
        for k in ("name", "pseudonym", "user_name", "userName"):
            cand = str(item.get(k) or "").strip()
            if _is_valid_profile_name(cand):
                detected_profile_name = cand
                break
        if detected_profile_name:
            break

    df = pd.DataFrame(raw_positions)

    def col(frame: pd.DataFrame, names: list[str], default=None) -> pd.Series:
        for n in names:
            if n in frame.columns:
                return frame[n]
        return pd.Series([default] * len(frame), index=frame.index)

    df["status"] = col(df, ["status", "_queried_status"], "CLOSED").astype(str).str.upper()
    df["condition_id"] = col(df, ["condition_id"], "").astype(str).str.strip()
    df["token_id"] = col(df, ["token_id", "asset", "asset_id"], "").astype(str).str.strip()
    df["title"] = col(df, ["title", "question"], "").astype(str).str.strip()
    df["outcome"] = col(df, ["outcome"], "").astype(str).str.strip()

    dedup_key = np.where(
        df["token_id"] != "",
        df["token_id"] + "_" + df["status"],
        df["condition_id"] + "_" + df["outcome"] + "_" + df["status"],
    )
    df["_dedup_key"] = dedup_key
    df = df.drop_duplicates(subset=["_dedup_key"], keep="first").drop(columns=["_dedup_key"]).reset_index(drop=True)

    df["avg_price"] = pd.to_numeric(col(df, ["avg_price"]), errors="coerce").fillna(0.0)
    df["entry_cost_usdc"] = pd.to_numeric(col(df, ["entry_cost_usdc"]), errors="coerce").fillna(0.0)
    df["current_size"] = pd.to_numeric(col(df, ["current_size"]), errors="coerce").fillna(0.0)
    df["current_value"] = pd.to_numeric(col(df, ["current_value"]), errors="coerce").fillna(0.0)

    realized_s = pd.to_numeric(col(df, ["realized_pnl"]), errors="coerce").fillna(0.0)
    unrealized_raw = pd.to_numeric(col(df, ["unrealized_pnl"]), errors="coerce")
    implied_unrealized = np.where(df["status"] == "OPEN", df["current_value"] - df["entry_cost_usdc"], 0.0)
    unrealized_s = unrealized_raw.fillna(pd.Series(implied_unrealized, index=df.index)).fillna(0.0)
    df["realized_pnl"] = realized_s
    df["unrealized_pnl"] = unrealized_s
    df["total_pnl"] = df["realized_pnl"] + df["unrealized_pnl"]

    df["pos_last_ts"] = col(df, ["last_event_at", "updated_at", "timestamp"]).apply(to_epoch)
    df["pos_first_ts"] = col(df, ["first_entry_at", "created_at"]).apply(to_epoch)

    if act_rows:
        if not detected_profile_name:
            for item in act_rows[:30]:
                for k in ("name", "pseudonym", "user_name", "userName"):
                    cand = str(item.get(k) or "").strip()
                    if _is_valid_profile_name(cand):
                        detected_profile_name = cand
                        break
                if detected_profile_name:
                    break

        adf = pd.DataFrame(act_rows)
        adf["act_type"] = col(adf, ["type", "activity_type"], "TRADE").astype(str).str.upper()
        adf["act_token"] = col(adf, ["token_id", "asset", "asset_id"], "").astype(str).str.strip()
        adf["act_cid"] = col(adf, ["condition_id"], "").astype(str).str.strip()
        adf["act_side"] = col(adf, ["side"], "").astype(str).str.upper()
        adf["act_price"] = pd.to_numeric(col(adf, ["price"]), errors="coerce").fillna(0.0)
        adf["act_size"] = pd.to_numeric(col(adf, ["size"]), errors="coerce").fillna(0.0)
        adf["act_usdc"] = pd.to_numeric(col(adf, ["usdc_size", "amount"]), errors="coerce").fillna(0.0)
        adf["act_usdc"] = np.where(adf["act_usdc"] > 0, adf["act_usdc"], adf["act_price"] * adf["act_size"])
        adf["act_ts"] = col(adf, ["timestamp", "created_at"]).apply(to_epoch)

        trades = adf[adf["act_type"] == "TRADE"].copy()
        valid_trade = trades[trades["act_token"] != ""]
        buys = valid_trade[valid_trade["act_side"] == "BUY"]
        sells = valid_trade[valid_trade["act_side"] == "SELL"]

        if not buys.empty:
            b_agg = buys.groupby("act_token", as_index=False).agg(
                act_buy_usdc=("act_usdc", "sum"),
                act_buy_shares=("act_size", "sum"),
                buy_count=("act_usdc", "count"),
                first_buy_ts=("act_ts", "min"),
                last_buy_ts=("act_ts", "max"),
            )
            df = df.merge(b_agg, left_on="token_id", right_on="act_token", how="left")
        else:
            df["act_buy_usdc"] = 0.0
            df["act_buy_shares"] = 0.0
            df["buy_count"] = 0
            df["first_buy_ts"] = np.nan
            df["last_buy_ts"] = np.nan

        if not sells.empty:
            s_agg = sells.groupby("act_token", as_index=False).agg(
                act_sell_usdc=("act_usdc", "sum"),
                sell_count=("act_usdc", "count"),
                last_sell_ts=("act_ts", "max"),
            )
            df = df.merge(s_agg, left_on="token_id", right_on="act_token", how="left")
        else:
            df["act_sell_usdc"] = 0.0
            df["sell_count"] = 0
            df["last_sell_ts"] = np.nan

        splits = adf[(adf["act_type"] == "SPLIT") & (adf["act_cid"] != "")]
        if not splits.empty:
            sp_agg = splits.groupby("act_cid", as_index=False).agg(
                split_usdc=("act_usdc", "sum"),
                split_shares=("act_size", "sum"),
                split_count=("act_usdc", "count"),
                first_split_ts=("act_ts", "min"),
            )
            df = df.merge(sp_agg, left_on="condition_id", right_on="act_cid", how="left")
        else:
            df["split_usdc"] = 0.0
            df["split_shares"] = 0.0
            df["split_count"] = 0
            df["first_split_ts"] = np.nan

        merges = adf[(adf["act_type"] == "MERGE") & (adf["act_cid"] != "")]
        if not merges.empty:
            mg_agg = merges.groupby("act_cid", as_index=False).agg(
                merge_usdc=("act_usdc", "sum"),
                merge_count=("act_usdc", "count"),
                last_merge_ts=("act_ts", "max"),
            )
            df = df.merge(mg_agg, left_on="condition_id", right_on="act_cid", how="left")
        else:
            df["merge_usdc"] = 0.0
            df["merge_count"] = 0
            df["last_merge_ts"] = np.nan
    else:
        for c in ("act_buy_usdc", "act_buy_shares", "act_sell_usdc", "split_usdc", "split_shares", "merge_usdc"):
            df[c] = 0.0
        for c in ("buy_count", "sell_count", "split_count", "merge_count"):
            df[c] = 0
        for c in ("first_buy_ts", "last_buy_ts", "last_sell_ts", "first_split_ts", "last_merge_ts"):
            df[c] = np.nan

    for c in ("buy_count", "sell_count", "split_count", "merge_count"):
        df[c] = df[c].fillna(0).astype(int)
    for c in ("act_buy_usdc", "act_buy_shares", "act_sell_usdc", "split_usdc", "split_shares", "merge_usdc"):
        df[c] = df[c].fillna(0.0)

    clob_cost = np.maximum(df["entry_cost_usdc"], df["act_buy_usdc"])
    split_side_cost = np.where(df["split_usdc"] > 0, df["split_usdc"] * 0.5, df["split_shares"] * 0.5)

    can_imply_win_cost = (
        (clob_cost <= 0.05)
        & (split_side_cost <= 0)
        & (df["avg_price"] > 0.001)
        & (df["avg_price"] < 0.999)
        & (df["realized_pnl"] > 0)
        & (df["merge_count"] == 0)
    )
    can_imply_loss_cost = (
        (clob_cost <= 0.05)
        & (split_side_cost <= 0)
        & (df["realized_pnl"] < 0)
        & (df["merge_count"] == 0)
    )
    implied_cost_val = np.where(
        can_imply_win_cost,
        df["realized_pnl"] * df["avg_price"] / (1.0 - df["avg_price"]),
        np.where(can_imply_loss_cost, df["realized_pnl"].abs(), 0.0),
    )

    df["true_cost"] = np.where(
        clob_cost > 0.05,
        clob_cost,
        np.where(
            split_side_cost > 0,
            split_side_cost,
            np.where(implied_cost_val > 0.05, implied_cost_val, clob_cost),
        ),
    )

    df["cost_source"] = np.where(
        clob_cost > 0.05,
        np.where(df["act_buy_usdc"] >= df["entry_cost_usdc"], "ACTIVITY_TRADE", "POSITIONS_API"),
        np.where(
            split_side_cost > 0,
            "ACTIVITY_SPLIT",
            np.where(implied_cost_val > 0.05, "IMPLIED_FROM_PNL", "UNAVAILABLE"),
        ),
    )

    implied_p = np.where(
        df["act_buy_shares"] > 0,
        df["act_buy_usdc"] / df["act_buy_shares"].replace(0, np.nan),
        np.where(df["split_count"] > 0, 0.5, 0.0),
    )
    df["avg_price"] = np.where(df["avg_price"] > 0, df["avg_price"], implied_p)

    non_dust_mask = (
        (df["status"] == "OPEN")
        | (df["total_pnl"].abs() > 1e-6)
        | (df["true_cost"] > 0.01)
        | (df["buy_count"] > 0)
        | (df["sell_count"] > 0)
        | (df["split_count"] > 0)
        | (df["merge_count"] > 0)
    )
    df = df[non_dust_mask].reset_index(drop=True)

    df["first_ts"] = (
        df["first_buy_ts"]
        .fillna(df["first_split_ts"])
        .fillna(df["pos_first_ts"])
        .fillna(df["pos_last_ts"])
    )
    df["last_ts"] = (
        df["pos_last_ts"]
        .fillna(df["last_merge_ts"])
        .fillna(df["last_sell_ts"])
        .fillna(df["last_buy_ts"])
        .fillna(df["first_ts"])
    )
    df["accum_hours"] = ((df["last_buy_ts"].fillna(df["first_ts"]) - df["first_ts"]) / 3600.0).clip(lower=0.0).round(2)
    df["hold_hours"] = ((df["last_ts"] - df["first_ts"]) / 3600.0).clip(lower=0.0).round(2)

    dt_first = pd.to_datetime(df["first_ts"], unit="s", utc=True, errors="coerce").dt.tz_convert(VN_TZ)
    dt_last = pd.to_datetime(df["last_ts"], unit="s", utc=True, errors="coerce").dt.tz_convert(VN_TZ)
    df["entry_time_vn"] = dt_first.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    df["last_time_vn"] = dt_last.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    df["month"] = dt_first.dt.strftime("%Y-%m").fillna("Unknown")

    is_open = df["status"] == "OPEN"
    has_split_merge = (df["split_count"] > 0) | (df["merge_count"] > 0) | (df["cost_source"] == "ACTIVITY_SPLIT")
    is_valid_clob_price = (
        (~is_open)
        & (~has_split_merge)
        & (df["cost_source"].isin(["POSITIONS_API", "ACTIVITY_TRADE"]))
        & (df["true_cost"] > 0.05)
        & (df["avg_price"] > 0.0)
        & (df["avg_price"] <= 1.0)
    )

    df["record_type"] = np.where(
        is_open,
        "OPEN",
        np.where(has_split_merge, "SETTLED_SPLIT_MERGE", np.where(is_valid_clob_price, "SETTLED_CLOB", "SETTLED_ESTIMATED")),
    )
    df["is_win"] = np.where(
        df["status"] == "REDEEMABLE",
        True,
        np.where(df["status"] == "REDEEMABLE_LOST", False, df["total_pnl"] > 0),
    )

    df["roi_pct"] = np.where(
        (df["true_cost"] > 0.05) & (df["cost_source"] != "UNAVAILABLE"),
        (df["total_pnl"] / df["true_cost"]) * 100.0,
        np.nan,
    ).round(2)

    df["avg_clip_usdc"] = np.where(df["buy_count"] > 0, df["true_cost"] / df["buy_count"], df["true_cost"]).round(2)
    df["exit_style"] = np.where(
        df["merge_count"] > 0,
        "MERGED_2_SIDES",
        np.where(
            df["split_count"] > 0,
            np.where(df["sell_count"] > 0, "SPLIT_SOLD_SIDE", "SPLIT_HOLD"),
            np.where(
                df["sell_count"] == 0,
                "HOLD_TO_EXPIRY",
                np.where(df["act_sell_usdc"] >= df["true_cost"] * 0.8, "SOLD_EARLY", "PARTIAL_SELL"),
            ),
        ),
    )

    cid_outcomes = df[df["condition_id"] != ""].groupby("condition_id")["outcome"].nunique()
    hedged_set = set(cid_outcomes[cid_outcomes >= 2].index)
    df["is_hedged"] = df["condition_id"].isin(hedged_set)
    df["market_type"] = df["title"].apply(parse_market_type)
    df["tournament"] = df["title"].apply(parse_tournament)

    df["avg_price"] = df["avg_price"].round(4)
    df["true_cost"] = df["true_cost"].round(2)
    df["realized_pnl"] = df["realized_pnl"].round(2)
    df["unrealized_pnl"] = df["unrealized_pnl"].round(2)
    df["total_pnl"] = df["total_pnl"].round(2)

    # Gộp 2 chân SPLIT/MERGE theo condition_id (Cách C)
    sm_mask = (df["record_type"] == "SETTLED_SPLIT_MERGE") & (df["condition_id"] != "")
    sm_df = df[sm_mask].copy()
    other_df = df[~sm_mask].copy()

    if not sm_df.empty:
        collapsed_rows = [
            _collapse_split_merge_group(grp)
            for _, grp in sm_df.groupby("condition_id", sort=False)
        ]
        df = pd.concat([other_df, pd.DataFrame(collapsed_rows)], ignore_index=True)
    else:
        df = other_df.reset_index(drop=True)

    df["is_backtest_clean"] = ~df["record_type"].isin(["OPEN", "SETTLED_ESTIMATED"])
    df["is_clob_only"] = df["record_type"] == "SETTLED_CLOB"
    df = df.sort_values("first_ts", ascending=True, na_position="last").reset_index(drop=True)

    cols = [
        "entry_time_vn", "last_time_vn", "month", "status",
        "record_type", "is_backtest_clean", "is_clob_only", "cost_source",
        "condition_id", "token_id", "title", "outcome",
        "market_type", "tournament", "is_hedged",
        "avg_price", "true_cost", "realized_pnl", "unrealized_pnl", "total_pnl", "roi_pct", "is_win",
        "buy_count", "sell_count", "split_count", "merge_count",
        "avg_clip_usdc", "accum_hours", "hold_hours", "exit_style",
    ]
    out_df = df[cols].copy()

    vol_1w = float(out_df["true_cost"].sum())
    pnl_net = float(out_df["total_pnl"].sum())
    mkt_cnt = int(out_df[out_df["condition_id"] != ""]["condition_id"].nunique())
    roi_val = (pnl_net / vol_1w * 100.0) if vol_1w > 0 else 0.0

    clean_sub = out_df[out_df["is_backtest_clean"]]
    clean_vol = float(clean_sub["true_cost"].sum())
    clean_pnl = float(clean_sub["total_pnl"].sum())
    clean_mkts = int(clean_sub[clean_sub["condition_id"] != ""]["condition_id"].nunique())
    clean_roi = (clean_pnl / clean_vol * 100.0) if clean_vol > 0 else 0.0
    clean_wr = float(clean_sub["is_win"].mean() * 100.0) if not clean_sub.empty else 0.0

    summary = {
        "volume": vol_1w, "pnl": pnl_net, "markets": mkt_cnt, "roi_pct": roi_val,
        "clean_volume": clean_vol, "clean_pnl": clean_pnl, "clean_markets": clean_mkts,
        "clean_roi_pct": clean_roi, "clean_winrate": clean_wr, "clean_trades": int(len(clean_sub)),
        "detected_profile_name": detected_profile_name,
    }
    return out_df, summary


# ======================== UI ========================
st.title("🧬 Polymarket Backtest Extractor")

col_w, col_n = st.columns([2.2, 1.3])
wallet_input = col_w.text_input("Địa chỉ ví (0x...) hoặc @username", value=DEFAULT_WALLET)
custom_name_input = col_n.text_input("Tên file gợi nhớ (tùy chọn)", placeholder="Bỏ trống = tự lấy tên profile")

opt_c1, opt_c2, opt_c3 = st.columns([1.2, 1.0, 1.0])
ai_clean_only = opt_c1.checkbox("Bản AI: Chỉ kèo đã tất toán", value=True)
ai_months_opt = opt_c2.selectbox(
    "Phạm vi tháng (AI)",
    options=[2, 1, 3, 0],
    format_func=lambda m: "Toàn bộ lịch sử" if m == 0 else f"{m} tháng gần nhất",
)
ai_min_cost = opt_c3.number_input("Min vốn ví gốc ($)", min_value=0, value=0, step=100)

if st.button("⚡ TRÍCH XUẤT CSV", type="primary", use_container_width=True):
    with st.spinner("Đang đồng bộ đa luồng & kiểm toán dữ liệu..."):
        try:
            resolved_wallet, preset_name, err_msg = resolve_wallet_only(wallet_input)
            if err_msg:
                st.error(err_msg)
                st.stop()

            ledger_df, summary = build_backtest_ledger(resolved_wallet)

            auto_name = preset_name
            if not auto_name:
                api_prof = sanitize_filename_part(summary.get("detected_profile_name", ""))
                if api_prof:
                    auto_name = api_prof
                else:
                    found_name = lookup_profile_name_by_wallet(resolved_wallet)
                    auto_name = sanitize_filename_part(found_name) or f"{resolved_wallet[:6]}_{resolved_wallet[-4:]}"
                st.session_state.setdefault("wallet_profile_cache", {})[wallet_input.strip().lower()] = (
                    resolved_wallet,
                    auto_name,
                )

            st.session_state["backtest_pack"] = {
                "ledger_df": ledger_df,
                "summary": summary,
                "resolved_wallet": resolved_wallet,
                "auto_name": auto_name,
            }
        except Exception as e:
            st.error(f"Lỗi gọi API: {e}")
            st.stop()

if "backtest_pack" in st.session_state:
    pack = st.session_state["backtest_pack"]
    ledger_df = pack["ledger_df"]
    summary = pack["summary"]
    resolved_wallet = pack["resolved_wallet"]
    auto_name = pack["auto_name"]

    if ledger_df.empty:
        st.warning(f"Không có dữ liệu cho ví `{resolved_wallet}`.")
    else:
        final_label = sanitize_filename_part(custom_name_input) or auto_name
        ai_df = build_ai_slim_csv(
            ledger_df,
            clean_only=ai_clean_only,
            recent_months=ai_months_opt,
            min_whale_cost=float(ai_min_cost),
        )

        raw_bytes = ledger_df.to_csv(index=False).encode("utf-8-sig")
        ai_bytes = ai_df.to_csv(index=False).encode("utf-8-sig")
        saved_pct = (1.0 - len(ai_bytes) / max(len(raw_bytes), 1)) * 100.0
        est_ai_tokens = int(len(ai_bytes) / 3.2)

        st.caption(
            f"Ví: `{resolved_wallet}` • Profile: **{auto_name}** • "
            f"Bản AI gọn hơn **{saved_pct:.0f}%** ({len(ai_df):,}/{len(ledger_df):,} dòng • ~**{est_ai_tokens:,} tokens**)"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volume (Mua 1 chiều)", f"${summary['volume']:,.0f}")
        c2.metric("PNL", f"${summary['pnl']:+,.2f}")
        c3.metric("Market", f"{summary['markets']:,}")
        c4.metric("ROI", f"{summary['roi_pct']:+.3f}%")

        st.info(
            f"📊 **Tập chuẩn Backtest (Cách C — `is_backtest_clean = True`):** "
            f"**{summary['clean_trades']:,}** lệnh ({summary['clean_markets']:,} markets) • "
            f"Winrate: **{summary['clean_winrate']:.2f}%** • "
            f"PNL: **${summary['clean_pnl']:+,.2f}** • "
            f"ROI: **{summary['clean_roi_pct']:+.3f}%**"
        )

        dl1, dl2 = st.columns(2)
        dl1.download_button(
            label=f"⚡ TẢI CSV SIÊU GỌN CHO AI ({final_label}_ai.csv)",
            data=ai_bytes,
            file_name=f"{final_label}_ai.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
            disabled=ai_df.empty,
        )
        dl2.download_button(
            label=f"📦 TẢI CSV GỐC ĐẦY ĐỦ ({final_label}.csv)",
            data=raw_bytes,
            file_name=f"{final_label}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("📋 Copy tóm tắt ~550 tokens (Dán hỏi Gemini nhanh không cần tải CSV)", expanded=True):
            digest_txt = build_ai_prompt_digest(ledger_df, resolved_wallet, final_label)
            st.code(digest_txt, language="text")
