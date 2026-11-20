import re
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Polymarket Backtest Extractor",
    page_icon="🧬",
    layout="centered",
)

BASE_V2_URL = "https://data-api.polymarket.com/v2"
POSITIONS_URL = f"{BASE_V2_URL}/positions"
ACTIVITY_URL = f"{BASE_V2_URL}/activity"
DEFAULT_WALLET = "0x474D6503CD4867D611b012ED6c8a6fAAaf5e6b5f"
DEFAULT_ACCOUNT = "HoangSon2"
VN_TZ = timezone(timedelta(hours=7))


def fetch_v2_cursor(endpoint: str, base_params: dict, max_pages: int = 150) -> list[dict]:
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


def build_csv_filename(wallet: str, account_name: str = "") -> str:
    """Trả về tên file theo công thức `<tên tài khoản>.csv` (vd: `hoangson2.csv`)."""
    raw = (account_name or "").strip().lstrip("@")
    if not raw:
        w = (wallet or "").strip()
        if w and not w.lower().startswith("0x"):
            raw = w.lstrip("@")
        elif w.lower().startswith("0x") and len(w) >= 10:
            raw = f"wallet_{w[2:10].lower()}"
        else:
            raw = "account"

    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_").lower()
    return f"{slug or 'account'}.csv"


def build_backtest_ledger(wallet: str) -> tuple[pd.DataFrame, dict]:
    """Trích xuất sổ cái chuẩn từ /v2/positions & /v2/activity làm nguồn dữ liệu phân tích."""
    w = wallet.strip()
    raw_positions = []
    for st_val in ("CLOSED", "REDEEMABLE", "REDEEMABLE_LOST", "MERGEABLE", "OPEN"):
        batch = fetch_v2_cursor(POSITIONS_URL, {"user": w, "status": st_val, "limit": 500}, max_pages=120)
        for item in batch:
            item["_queried_status"] = st_val
        raw_positions.extend(batch)

    if not raw_positions:
        return pd.DataFrame(), {"volume": 0.0, "pnl": 0.0, "markets": 0, "roi_pct": 0.0}

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

    act_rows = fetch_v2_cursor(ACTIVITY_URL, {"user": w, "limit": 500}, max_pages=150)
    if act_rows:
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
    df["is_backtest_clean"] = is_valid_clob_price
    df["is_win"] = df["total_pnl"] > 0

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

    df = df.sort_values("first_ts", ascending=True, na_position="last").reset_index(drop=True)

    cols = [
        "entry_time_vn", "last_time_vn", "month", "status",
        "record_type", "is_backtest_clean", "cost_source",
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

    summary = {
        "volume": vol_1w,
        "pnl": pnl_net,
        "markets": mkt_cnt,
        "roi_pct": roi_val,
    }
    return out_df, summary


# ======================== MINIMAL 4-METRIC BACKTEST UI ========================
st.title("🧬 Polymarket Backtest Extractor")

wallet_input = st.text_input("Địa chỉ ví", value=DEFAULT_WALLET)
account_input = st.text_input("Tên tài khoản", value=DEFAULT_ACCOUNT)

if st.button("⚡ TRÍCH XUẤT CSV", type="primary", use_container_width=True):
    with st.spinner("Đang đồng bộ dữ liệu..."):
        try:
            ledger_df, summary = build_backtest_ledger(wallet_input)
            csv_filename = build_csv_filename(wallet_input, account_input)
            st.session_state["backtest_pack"] = (ledger_df, summary, csv_filename)
        except Exception as e:
            st.error(f"Lỗi gọi API: {e}")
            st.stop()

if "backtest_pack" in st.session_state:
    ledger_df, summary, csv_filename = st.session_state["backtest_pack"]
    if ledger_df.empty:
        st.warning("Không có dữ liệu cho ví này.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volume", f"${summary['volume']:,.0f}")
        c2.metric("PNL", f"${summary['pnl']:+,.2f}")
        c3.metric("Market", f"{summary['markets']:,}")
        c4.metric("ROI", f"{summary['roi_pct']:+.3f}%")

        st.download_button(
            label=f"📥 TẢI FILE CSV ({csv_filename})",
            data=ledger_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=csv_filename,
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
