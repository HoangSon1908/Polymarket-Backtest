import re
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Polymarket AI Data Extractor",
    page_icon="🧬",
    layout="centered",
)

BASE_V2_URL = "https://data-api.polymarket.com/v2"
POSITIONS_URL = f"{BASE_V2_URL}/positions"
ACTIVITY_URL = f"{BASE_V2_URL}/activity"
DEFAULT_WALLET = "0xc9bfc7986767173b4f1c3fc9b2c6b255fc7381c0"
VN_TZ = timezone(timedelta(hours=7))


def fetch_v2_cursor(endpoint: str, base_params: dict, max_pages: int = 80) -> list[dict]:
    """Kéo toàn bộ dữ liệu theo chuẩn Polymarket Data API v2 (Cursor-only, không dùng offset)."""
    rows = []
    params = dict(base_params)
    params["limit"] = min(int(params.get("limit", 500)), 500)
    anchor_keys = [k for k in ("user", "status", "type", "side") if k in params]

    for _ in range(max_pages):
        res = requests.get(endpoint, params=params, timeout=20)
        if res.status_code == 429:
            time.sleep(2.0)
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

        if not next_cursor or not has_more:
            break

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


def build_ai_dataset(wallet: str) -> pd.DataFrame:
    """Trích xuất & làm giàu dữ liệu nguyên tử từ /v2/positions và /v2/activity."""
    raw_positions = []
    for st_val in ("CLOSED", "REDEEMABLE", "REDEEMABLE_LOST"):
        batch = fetch_v2_cursor(POSITIONS_URL, {"user": wallet.strip(), "status": st_val, "limit": 500})
        for item in batch:
            item["_queried_status"] = st_val
        raw_positions.extend(batch)

    if not raw_positions:
        return pd.DataFrame()

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

    df["avg_price"] = pd.to_numeric(col(df, ["avg_price"]), errors="coerce").fillna(0.0)
    df["entry_cost_usdc"] = pd.to_numeric(col(df, ["entry_cost_usdc"]), errors="coerce").fillna(0.0)

    # Chuẩn Data API v2: total_pnl = realized_pnl + unrealized_pnl
    realized_s = pd.to_numeric(col(df, ["realized_pnl"]), errors="coerce").fillna(0.0)
    unrealized_s = pd.to_numeric(col(df, ["unrealized_pnl"]), errors="coerce").fillna(0.0)
    df["total_pnl"] = realized_s + unrealized_s

    df["pos_last_ts"] = col(df, ["last_event_at", "updated_at", "timestamp"]).apply(to_epoch)
    df["pos_first_ts"] = col(df, ["first_entry_at", "created_at"]).apply(to_epoch)

    # Kéo /v2/activity để tái tạo hành vi gom lệnh & thoát lệnh
    act_rows = fetch_v2_cursor(ACTIVITY_URL, {"user": wallet.strip(), "type": "TRADE", "limit": 500}, max_pages=60)
    if act_rows:
        adf = pd.DataFrame(act_rows)
        adf["act_token"] = col(adf, ["token_id", "asset", "asset_id"], "").astype(str).str.strip()
        adf["act_side"] = col(adf, ["side"], "").astype(str).str.upper()
        adf["act_price"] = pd.to_numeric(col(adf, ["price"]), errors="coerce").fillna(0.0)
        adf["act_size"] = pd.to_numeric(col(adf, ["size"]), errors="coerce").fillna(0.0)
        adf["act_usdc"] = pd.to_numeric(col(adf, ["usdc_size", "amount"]), errors="coerce").fillna(0.0)
        adf["act_usdc"] = np.where(adf["act_usdc"] > 0, adf["act_usdc"], adf["act_price"] * adf["act_size"])
        adf["act_ts"] = col(adf, ["timestamp", "created_at"]).apply(to_epoch)

        valid_act = adf[adf["act_token"] != ""].copy()
        buys = valid_act[valid_act["act_side"] == "BUY"]
        sells = valid_act[valid_act["act_side"] == "SELL"]

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
    else:
        df["act_buy_usdc"] = 0.0
        df["act_buy_shares"] = 0.0
        df["buy_count"] = 0
        df["first_buy_ts"] = np.nan
        df["last_buy_ts"] = np.nan
        df["act_sell_usdc"] = 0.0
        df["sell_count"] = 0
        df["last_sell_ts"] = np.nan

    df["buy_count"] = df["buy_count"].fillna(0).astype(int)
    df["sell_count"] = df["sell_count"].fillna(0).astype(int)
    df["act_buy_usdc"] = df["act_buy_usdc"].fillna(0.0)
    df["act_sell_usdc"] = df["act_sell_usdc"].fillna(0.0)

    # Khôi phục vốn gốc & giá vào trung bình chuẩn xác
    df["true_cost"] = np.where(df["entry_cost_usdc"] > 0, df["entry_cost_usdc"], df["act_buy_usdc"])
    implied_p = np.where(df["act_buy_shares"].fillna(0) > 0, df["act_buy_usdc"] / df["act_buy_shares"].replace(0, np.nan), 0.0)
    df["avg_price"] = np.where(df["avg_price"] > 0, df["avg_price"], implied_p)

    # Lọc bỏ bản ghi lỗi/rỗng
    df = df[(df["true_cost"] > 0.01) & (df["avg_price"] > 0) & (df["avg_price"] < 1.0)].copy()

    # Thời gian & thời lượng
    df["first_ts"] = df["first_buy_ts"].fillna(df["pos_first_ts"]).fillna(df["pos_last_ts"])
    df["last_ts"] = df["pos_last_ts"].fillna(df["last_sell_ts"]).fillna(df["last_buy_ts"]).fillna(df["first_ts"])
    df["accum_hours"] = ((df["last_buy_ts"].fillna(df["first_ts"]) - df["first_ts"]) / 3600.0).clip(lower=0.0).round(2)
    df["hold_hours"] = ((df["last_ts"] - df["first_ts"]) / 3600.0).clip(lower=0.0).round(2)

    dt_first = pd.to_datetime(df["first_ts"], unit="s", utc=True, errors="coerce").dt.tz_convert(VN_TZ)
    dt_last = pd.to_datetime(df["last_ts"], unit="s", utc=True, errors="coerce").dt.tz_convert(VN_TZ)
    df["entry_time_vn"] = dt_first.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    df["last_time_vn"] = dt_last.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    df["month"] = dt_first.dt.strftime("%Y-%m").fillna("Unknown")

    # Kết quả & chỉ số tài chính
    df["is_win"] = np.where(
        df["status"] == "REDEEMABLE",
        True,
        np.where(df["status"] == "REDEEMABLE_LOST", False, df["total_pnl"] > 0),
    )
    df["roi_pct"] = ((df["total_pnl"] / df["true_cost"]) * 100.0).round(2)
    df["avg_clip_usdc"] = np.where(df["buy_count"] > 0, df["true_cost"] / df["buy_count"], df["true_cost"]).round(2)
    df["exit_style"] = np.where(
        df["sell_count"] == 0,
        "HOLD_TO_EXPIRY",
        np.where(df["act_sell_usdc"] >= df["true_cost"] * 0.8, "SOLD_EARLY", "PARTIAL_SELL"),
    )

    # Cờ phát hiện Hedge 2 đầu trên cùng condition_id
    cid_outcomes = df[df["condition_id"] != ""].groupby("condition_id")["outcome"].nunique()
    hedged_set = set(cid_outcomes[cid_outcomes >= 2].index)
    df["is_hedged"] = df["condition_id"].isin(hedged_set)

    # Bóc tách đặc trưng CS2
    df["market_type"] = df["title"].apply(parse_market_type)
    df["tournament"] = df["title"].apply(parse_tournament)

    # Làm tròn chuẩn số học để file CSV gọn nhẹ
    df["avg_price"] = df["avg_price"].round(4)
    df["true_cost"] = df["true_cost"].round(2)
    df["total_pnl"] = df["total_pnl"].round(2)

    df = df.sort_values("first_ts", ascending=True, na_position="last").reset_index(drop=True)

    final_cols = [
        "entry_time_vn", "last_time_vn", "month", "status",
        "condition_id", "token_id", "title", "outcome",
        "market_type", "tournament", "is_hedged",
        "avg_price", "true_cost", "total_pnl", "roi_pct", "is_win",
        "buy_count", "sell_count", "avg_clip_usdc",
        "accum_hours", "hold_hours", "exit_style",
    ]
    return df[final_cols]


# ======================== MINIMAL EXTRACTOR UI ========================
st.title("🧬 Polymarket AI Data Extractor")
st.caption("Trích xuất dữ liệu nguyên tử chuẩn hóa từ API v2 để Gemini phân tích trực tiếp.")

wallet_input = st.text_input("Địa chỉ ví cần trích xuất", value=DEFAULT_WALLET)

if st.button("⚡ TRÍCH XUẤT DỮ LIỆU & TẠO FILE CSV", type="primary", use_container_width=True):
    with st.spinner("Đang đồng bộ /v2/positions & /v2/activity..."):
        try:
            clean_df = build_ai_dataset(wallet_input)
            now_vn = datetime.now(VN_TZ).strftime("%Y%m%d_%H%M")
            st.session_state["ai_csv_pack"] = (clean_df, now_vn)
        except Exception as e:
            st.error(f"Lỗi gọi API Polymarket v2: {e}")
            st.stop()

if "ai_csv_pack" in st.session_state:
    clean_df, ts_tag = st.session_state["ai_csv_pack"]
    if clean_df.empty:
        st.warning("Không có dữ liệu vị thế đã đóng nào cho ví này.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số vị thế sạch", f"{len(clean_df):,}")
        c2.metric("Khớp Activity", f"{(clean_df['buy_count'] > 0).mean()*100:.0f}%")
        c3.metric("Tổng vốn gốc", f"${clean_df['true_cost'].sum():,.0f}")
        c4.metric("PnL ròng", f"${clean_df['total_pnl'].sum():+,.0f}")

        csv_bytes = clean_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label=f"📥 TẢI FILE CSV CHO AI PHÂN TÍCH (polymarket_ai_{ts_tag}.csv)",
            data=csv_bytes,
            file_name=f"polymarket_ai_{ts_tag}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

        st.dataframe(clean_df.tail(15), use_container_width=True, hide_index=True)
