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
USER_STATS_URL = f"{BASE_V2_URL}/user-stats"
USER_VOLUME_URL = f"{BASE_V2_URL}/user-volume"
DEFAULT_WALLET = "0xc9bfc7986767173b4f1c3fc9b2c6b255fc7381c0"
VN_TZ = timezone(timedelta(hours=7))


def fetch_v2_cursor(endpoint: str, base_params: dict, max_pages: int = 120) -> list[dict]:
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


def fetch_optional_user_summary(wallet: str) -> dict:
    """Lấy thêm thống kê tổng hợp từ /v2/user-stats & /v2/user-volume nếu khả dụng."""
    summary = {}
    w = wallet.strip()
    for key, url in (("stats", USER_STATS_URL), ("volume", USER_VOLUME_URL)):
        try:
            res = requests.get(url, params={"user": w}, timeout=10)
            if res.status_code == 200:
                payload = res.json()
                summary[key] = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
        except Exception:
            pass
    return summary


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


def build_ai_dataset(wallet: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Trích xuất & đối soát dữ liệu nguyên tử từ /v2/positions (đủ 5 trạng thái)
    và /v2/activity (đủ TRADE, SPLIT, MERGE, REDEEM).
    Trả về: (clean_df, full_ledger_df, audit_meta)
    """
    w = wallet.strip()
    raw_positions = []
    all_statuses = ("CLOSED", "REDEEMABLE", "REDEEMABLE_LOST", "MERGEABLE", "OPEN")
    for st_val in all_statuses:
        batch = fetch_v2_cursor(POSITIONS_URL, {"user": w, "status": st_val, "limit": 500}, max_pages=100)
        for item in batch:
            item["_queried_status"] = st_val
        raw_positions.extend(batch)

    if not raw_positions:
        return pd.DataFrame(), pd.DataFrame(), {}

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

    # Khử trùng lặp nếu API trả cùng vị thế ở 2 lần truy vấn status
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

    # Chuẩn Data API v2: total_pnl = realized_pnl + unrealized_pnl
    realized_s = pd.to_numeric(col(df, ["realized_pnl"]), errors="coerce").fillna(0.0)
    unrealized_raw = pd.to_numeric(col(df, ["unrealized_pnl"]), errors="coerce")
    implied_unrealized = np.where(
        df["status"] == "OPEN",
        df["current_value"] - df["entry_cost_usdc"],
        0.0,
    )
    unrealized_s = unrealized_raw.fillna(pd.Series(implied_unrealized, index=df.index)).fillna(0.0)
    df["realized_pnl"] = realized_s
    df["unrealized_pnl"] = unrealized_s
    df["total_pnl"] = df["realized_pnl"] + df["unrealized_pnl"]

    df["pos_last_ts"] = col(df, ["last_event_at", "updated_at", "timestamp"]).apply(to_epoch)
    df["pos_first_ts"] = col(df, ["first_entry_at", "created_at"]).apply(to_epoch)

    # Kéo TOÀN BỘ /v2/activity (không giới hạn type="TRADE") để bắt cả SPLIT, MERGE, REDEEM
    act_rows = fetch_v2_cursor(ACTIVITY_URL, {"user": w, "limit": 500}, max_pages=120)

    total_two_way_vol = 0.0
    activity_type_counts = {}

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

        activity_type_counts = adf["act_type"].value_counts().to_dict()
        total_two_way_vol = float(adf["act_usdc"].sum())

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

        # Gom SPLIT & MERGE theo condition_id để nhận diện kèo mint/gộp 2 đầu
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

    # Khôi phục vốn gốc (ưu tiên entry_cost_usdc -> BUY trên CLOB -> 50% vốn SPLIT mỗi cửa)
    split_side_cost = np.where(df["split_usdc"] > 0, df["split_usdc"] * 0.5, df["split_shares"] * 0.5)
    df["true_cost"] = np.where(
        df["entry_cost_usdc"] > 0,
        df["entry_cost_usdc"],
        np.where(df["act_buy_usdc"] > 0, df["act_buy_usdc"], split_side_cost),
    )

    implied_p = np.where(
        df["act_buy_shares"] > 0,
        df["act_buy_usdc"] / df["act_buy_shares"].replace(0, np.nan),
        np.where(df["split_count"] > 0, 0.5, 0.0),
    )
    df["avg_price"] = np.where(df["avg_price"] > 0, df["avg_price"], implied_p)

    # Thời gian & thời lượng
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

    # Phân tầng bản ghi để không làm mất PnL toàn ví mà vẫn giữ CSV sạch cho AI
    is_open = df["status"] == "OPEN"
    is_valid_clean = (
        (~is_open)
        & (df["true_cost"] > 0.01)
        & (df["avg_price"] > 0.0)
        & (df["avg_price"] <= 1.0)
    )
    is_has_pnl_or_cost = (df["total_pnl"].abs() > 1e-6) | (df["true_cost"] > 0.01) | (df["merge_count"] > 0)

    df["record_group"] = np.where(
        is_open,
        "OPEN_ACTIVE",
        np.where(
            is_valid_clean,
            "CLEAN_SETTLED",
            np.where(is_has_pnl_or_cost, "MERGED_OR_UNMATCHED", "DUST_EMPTY"),
        ),
    )

    # Kết quả & chỉ số tài chính
    df["is_win"] = np.where(
        df["status"] == "REDEEMABLE",
        True,
        np.where(df["status"] == "REDEEMABLE_LOST", False, df["total_pnl"] > 0),
    )
    df["roi_pct"] = np.where(df["true_cost"] > 0.01, (df["total_pnl"] / df["true_cost"]) * 100.0, 0.0).round(2)
    df["avg_clip_usdc"] = np.where(df["buy_count"] > 0, df["true_cost"] / df["buy_count"], df["true_cost"]).round(2)
    df["exit_style"] = np.where(
        df["merge_count"] > 0,
        "MERGED_2_SIDES",
        np.where(
            df["sell_count"] == 0,
            "HOLD_TO_EXPIRY",
            np.where(df["act_sell_usdc"] >= df["true_cost"] * 0.8, "SOLD_EARLY", "PARTIAL_SELL"),
        ),
    )

    # Cờ phát hiện Hedge 2 đầu trên cùng condition_id
    cid_outcomes = df[df["condition_id"] != ""].groupby("condition_id")["outcome"].nunique()
    hedged_set = set(cid_outcomes[cid_outcomes >= 2].index)
    df["is_hedged"] = df["condition_id"].isin(hedged_set)

    # Bóc tách đặc trưng thị trường
    df["market_type"] = df["title"].apply(parse_market_type)
    df["tournament"] = df["title"].apply(parse_tournament)

    # Làm tròn chuẩn số học
    df["avg_price"] = df["avg_price"].round(4)
    df["true_cost"] = df["true_cost"].round(2)
    df["total_pnl"] = df["total_pnl"].round(2)

    df = df.sort_values("first_ts", ascending=True, na_position="last").reset_index(drop=True)

    clean_cols = [
        "entry_time_vn", "last_time_vn", "month", "status",
        "condition_id", "token_id", "title", "outcome",
        "market_type", "tournament", "is_hedged",
        "avg_price", "true_cost", "total_pnl", "roi_pct", "is_win",
        "buy_count", "sell_count", "avg_clip_usdc",
        "accum_hours", "hold_hours", "exit_style",
    ]
    full_cols = ["record_group"] + clean_cols + ["merge_count", "split_count", "realized_pnl", "unrealized_pnl"]

    clean_df = df[df["record_group"] == "CLEAN_SETTLED"][clean_cols].reset_index(drop=True)
    full_ledger_df = df[df["record_group"] != "DUST_EMPTY"][full_cols].reset_index(drop=True)

    # Tổng hợp số liệu đối soát toàn ví
    non_dust = df[df["record_group"] != "DUST_EMPTY"]
    audit_meta = {
        "total_unique_markets": int(non_dust[non_dust["condition_id"] != ""]["condition_id"].nunique()),
        "clean_unique_markets": int(clean_df[clean_df["condition_id"] != ""]["condition_id"].nunique()),
        "clean_pnl": float(clean_df["total_pnl"].sum()),
        "merged_unmatched_pnl": float(df[df["record_group"] == "MERGED_OR_UNMATCHED"]["total_pnl"].sum()),
        "open_pnl": float(df[df["record_group"] == "OPEN_ACTIVE"]["total_pnl"].sum()),
        "wallet_total_pnl": float(non_dust["total_pnl"].sum()),
        "clean_one_way_cost": float(clean_df["true_cost"].sum()),
        "two_way_activity_vol": total_two_way_vol,
        "activity_type_counts": activity_type_counts,
        "status_counts": non_dust["status"].value_counts().to_dict(),
    }
    return clean_df, full_ledger_df, audit_meta


# ======================== MINIMAL EXTRACTOR UI ========================
st.title("🧬 Polymarket AI Data Extractor")
st.caption("Trích xuất dữ liệu nguyên tử chuẩn hóa từ API v2 & đối soát sổ cái toàn ví.")

wallet_input = st.text_input("Địa chỉ ví cần trích xuất", value=DEFAULT_WALLET)

if st.button("⚡ TRÍCH XUẤT DỮ LIỆU & TẠO FILE CSV", type="primary", use_container_width=True):
    with st.spinner("Đang đồng bộ đủ 5 trạng thái /v2/positions & toàn bộ /v2/activity..."):
        try:
            clean_df, full_df, audit_meta = build_ai_dataset(wallet_input)
            user_summary = fetch_optional_user_summary(wallet_input)
            now_vn = datetime.now(VN_TZ).strftime("%Y%m%d_%H%M")
            st.session_state["ai_csv_pack"] = (clean_df, full_df, audit_meta, user_summary, now_vn)
        except Exception as e:
            st.error(f"Lỗi gọi API Polymarket v2: {e}")
            st.stop()

if "ai_csv_pack" in st.session_state:
    clean_df, full_df, audit_meta, user_summary, ts_tag = st.session_state["ai_csv_pack"]
    if full_df.empty:
        st.warning("Không có dữ liệu vị thế nào cho ví này.")
    else:
        extra_pnl = audit_meta["merged_unmatched_pnl"] + audit_meta["open_pnl"]
        act_match_pct = (clean_df["buy_count"] > 0).mean() * 100 if not clean_df.empty else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Vị thế sạch (CSV)",
            f"{len(clean_df):,}",
            f"Tổng: {audit_meta['total_unique_markets']:,} markets",
        )
        c2.metric(
            "Vốn gốc 1 chiều",
            f"${audit_meta['clean_one_way_cost']:,.0f}",
            f"Vol 2 chiều: ${audit_meta['two_way_activity_vol']:,.0f}",
        )
        c3.metric(
            "PnL Tập Sạch (CSV)",
            f"${audit_meta['clean_pnl']:+,.2f}",
            f"Khớp Activity: {act_match_pct:.0f}%",
        )
        c4.metric(
            "PnL Gộp Toàn Ví",
            f"${audit_meta['wallet_total_pnl']:+,.2f}",
            f"Ngoài CSV: ${extra_pnl:+,.2f}",
        )

        # Bảng đối soát minh bạch giữa File CSV sạch và Sổ cái toàn ví
        with st.expander("🔍 Bảng đối soát chi tiết từng nhóm vị thế (Đối chiếu với Polymarket UI)", expanded=True):
            recon_rows = [
                {
                    "Nhóm dữ liệu": "1. CLEAN_SETTLED (Xuất ra CSV cho AI)",
                    "Số dòng": len(clean_df),
                    "Số Market (condition_id)": audit_meta["clean_unique_markets"],
                    "Vốn gốc ($)": round(audit_meta["clean_one_way_cost"], 2),
                    "PnL ròng ($)": round(audit_meta["clean_pnl"], 2),
                },
                {
                    "Nhóm dữ liệu": "2. MERGED_OR_UNMATCHED (Kèo Merge 2 đầu / Ngoài khung Activity)",
                    "Số dòng": int((full_df["record_group"] == "MERGED_OR_UNMATCHED").sum()),
                    "Số Market (condition_id)": int(
                        full_df[full_df["record_group"] == "MERGED_OR_UNMATCHED"]["condition_id"].nunique()
                    ),
                    "Vốn gốc ($)": round(
                        float(full_df[full_df["record_group"] == "MERGED_OR_UNMATCHED"]["true_cost"].sum()), 2
                    ),
                    "PnL ròng ($)": round(audit_meta["merged_unmatched_pnl"], 2),
                },
                {
                    "Nhóm dữ liệu": "3. OPEN_ACTIVE (Đang mở / Chưa tất toán)",
                    "Số dòng": int((full_df["record_group"] == "OPEN_ACTIVE").sum()),
                    "Số Market (condition_id)": int(
                        full_df[full_df["record_group"] == "OPEN_ACTIVE"]["condition_id"].nunique()
                    ),
                    "Vốn gốc ($)": round(
                        float(full_df[full_df["record_group"] == "OPEN_ACTIVE"]["true_cost"].sum()), 2
                    ),
                    "PnL ròng ($)": round(audit_meta["open_pnl"], 2),
                },
            ]
            st.dataframe(pd.DataFrame(recon_rows), use_container_width=True, hide_index=True)
            st.caption(
                f"Phân bổ trạng thái API v2: `{audit_meta['status_counts']}` | "
                f"Loại sự kiện Activity: `{audit_meta['activity_type_counts']}`"
            )

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            csv_bytes = clean_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label=f"📥 TẢI CSV SẠCH CHO AI (polymarket_ai_{ts_tag}.csv)",
                data=csv_bytes,
                file_name=f"polymarket_ai_{ts_tag}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
        with dl_col2:
            full_csv_bytes = full_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label=f"📦 TẢI SỔ CÁI ĐẦY ĐỦ (gồm Merge & Open)",
                data=full_csv_bytes,
                file_name=f"polymarket_full_ledger_{ts_tag}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.dataframe(clean_df.tail(15), use_container_width=True, hide_index=True)
