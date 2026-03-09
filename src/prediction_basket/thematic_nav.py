from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThematicNavConfig:
    initial_capital: float = 100.0
    annual_risk_free_rate: float = 0.0
    fee_bps: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    force_exit_price_low: float = 0.05
    force_exit_price_high: float = 0.95
    roll_dte_days: int = 0
    target_invested_weight: float = 0.95
    cash_floor: float = 0.02
    cash_cap: float = 0.10
    immediate_redeploy: bool = True
    redistribute_unfilled_weight: bool = True
    settle_to_binary: bool = True
    spot_stale_days: int = 3


def load_selected_prices(processed_dir: str | Path, market_ids: list[str] | set[str] | None = None) -> pd.DataFrame:
    processed = Path(processed_dir)
    prices_path = processed / "prices.parquet"
    if not prices_path.exists():
        return pd.DataFrame(columns=["market_id", "date", "close_price", "volume"])

    prices = pd.read_parquet(prices_path).copy()
    if prices.empty:
        return prices
    prices["market_id"] = prices["market_id"].astype(str)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["close_price"] = pd.to_numeric(prices["close_price"], errors="coerce")
    prices = prices.dropna(subset=["market_id", "date", "close_price"]).copy()
    prices = prices[(prices["close_price"] >= 0.0) & (prices["close_price"] <= 1.0)].copy()
    if market_ids is not None:
        mids = {str(x) for x in market_ids if str(x)}
        prices = prices[prices["market_id"].isin(mids)].copy()
    prices = prices.sort_values(["market_id", "date"]).drop_duplicates(["market_id", "date"], keep="last")
    return prices.reset_index(drop=True)


def load_market_lifecycle(processed_dir: str | Path, market_ids: list[str] | set[str] | None = None) -> pd.DataFrame:
    processed = Path(processed_dir)
    history_path = processed / "polymarket_market_history.parquet"
    if not history_path.exists():
        return pd.DataFrame(
            columns=["market_id", "inactive_date", "created_date", "end_date", "resolution_date", "resolution_value"]
        )

    hist = pd.read_parquet(history_path).copy()
    hist["market_id"] = hist["market_id"].astype(str)
    if market_ids is not None:
        mids = {str(x) for x in market_ids if str(x)}
        hist = hist[hist["market_id"].isin(mids)].copy()

    def _to_naive_date(series: pd.Series) -> pd.Series:
        dt = pd.to_datetime(series, errors="coerce", utc=True)
        return dt.dt.tz_convert(None).dt.normalize()

    hist["created_date"] = _to_naive_date(hist["created_at"])
    hist["end_date"] = _to_naive_date(hist["end_date"])
    hist["closed_date"] = _to_naive_date(hist["closed_time"])
    hist["resolution_date"] = _to_naive_date(hist["resolution_time"])
    if "resolution_value" in hist.columns:
        hist["resolution_value"] = pd.to_numeric(hist["resolution_value"], errors="coerce")
    elif "winning_side" in hist.columns:
        side = hist["winning_side"].astype(str).str.strip().str.lower()
        hist["resolution_value"] = np.select(
            [
                side.isin({"yes", "y", "1", "true", "resolved_yes"}),
                side.isin({"no", "n", "0", "false", "resolved_no"}),
            ],
            [1.0, 0.0],
            default=np.nan,
        )
    else:
        hist["resolution_value"] = np.nan
    hist["inactive_date"] = pd.concat([hist["resolution_date"], hist["closed_date"], hist["end_date"]], axis=1).min(axis=1)
    hist = hist.sort_values(["market_id", "inactive_date", "created_date"]).drop_duplicates("market_id", keep="last")
    return hist[["market_id", "created_date", "end_date", "inactive_date", "resolution_date", "resolution_value"]].reset_index(drop=True)


def attach_asof_prices(compositions: pd.DataFrame, processed_dir: str | Path) -> pd.DataFrame:
    if compositions.empty:
        return compositions.copy()

    c = compositions.copy()
    drop_cols = [col for col in c.columns if col != "market_id" and col.startswith("market_id_")]
    if drop_cols:
        c = c.drop(columns=drop_cols, errors="ignore")
    c["rebalance_date"] = pd.to_datetime(c["rebalance_date"], errors="coerce")
    non_cash = c[~c["is_cash"]].copy()
    if non_cash.empty:
        c["effective_price"] = np.where(c["is_cash"], 1.0, np.nan)
        c["market_yes_price"] = np.where(c["is_cash"], np.nan, np.nan)
        c["effective_risk_price"] = c["effective_price"]
        return c

    mids = sorted(non_cash["market_id"].astype(str).unique())
    prices = load_selected_prices(processed_dir, mids)
    if prices.empty:
        if "effective_price" not in c.columns:
            c["effective_price"] = np.where(c["is_cash"], 1.0, np.nan)
        return c

    lookup = non_cash[["market_id", "rebalance_date"]].copy()
    lookup["market_id"] = lookup["market_id"].astype(str)
    lookup = lookup.drop_duplicates().sort_values(["market_id", "rebalance_date"]).reset_index(drop=True)
    px = prices[["market_id", "date", "close_price"]].sort_values(["market_id", "date"]).reset_index(drop=True)

    merged_parts: list[pd.DataFrame] = []
    for market_id, left_group in lookup.groupby("market_id", sort=False):
        right_group = px[px["market_id"] == market_id][["date", "close_price"]].copy()
        if right_group.empty:
            part = left_group.copy()
            part["asof_yes_price"] = np.nan
        else:
            part = pd.merge_asof(
                left_group.sort_values("rebalance_date"),
                right_group.sort_values("date"),
                left_on="rebalance_date",
                right_on="date",
                direction="backward",
                allow_exact_matches=True,
            )
            part["market_id"] = market_id
            part = part.rename(columns={"close_price": "asof_yes_price"}).drop(columns=["date"], errors="ignore")
        merged_parts.append(part)

    merged = pd.concat(merged_parts, ignore_index=True) if merged_parts else lookup.assign(asof_yes_price=np.nan)
    non_cash = non_cash.merge(merged, on=["market_id", "rebalance_date"], how="left")
    non_cash["current_price"] = pd.to_numeric(non_cash["asof_yes_price"], errors="coerce")
    non_cash["market_yes_price"] = non_cash["current_price"]
    side = non_cash.get("position_side", "YES").fillna("YES").astype(str)
    non_cash["effective_price"] = np.where(side.eq("NO"), 1.0 - non_cash["current_price"], non_cash["current_price"])
    non_cash["effective_risk_price"] = non_cash["effective_price"]
    if "spot_horizon_days" in non_cash.columns:
        non_cash["spot_effective_price_raw"] = non_cash["effective_risk_price"]
        non_cash["spot_effective_price_horizon"] = non_cash.apply(
            lambda row: _horizon_normalized_probability(
                row.get("effective_risk_price"),
                row.get("days_to_expiry"),
                row.get("spot_horizon_days"),
            ),
            axis=1,
        )
    non_cash = non_cash.drop(columns=["asof_yes_price"], errors="ignore")

    cash = c[c["is_cash"]].copy()
    cash["current_price"] = np.nan
    cash["market_yes_price"] = np.nan
    cash["effective_price"] = 1.0
    cash["effective_risk_price"] = 1.0
    if "spot_horizon_days" in cash.columns:
        cash["spot_effective_price_raw"] = 1.0
        cash["spot_effective_price_horizon"] = 1.0
    out = pd.concat([non_cash, cash], ignore_index=True)
    out = out.sort_values(["rebalance_date", "basket_code", "is_cash", "target_weight"], ascending=[True, True, True, False])
    return out.reset_index(drop=True)


def _effective_price_from_row(yes_price: float | int | None, side: str) -> float:
    p = pd.to_numeric(yes_price, errors="coerce")
    if pd.isna(p):
        return np.nan
    p = float(p)
    if side == "NO":
        return float(1.0 - p)
    return float(p)


def _horizon_normalized_probability(effective_price: float | int | None, days_to_expiry: float | int | None, horizon_days: float | int | None) -> float:
    p = pd.to_numeric(effective_price, errors="coerce")
    d = pd.to_numeric(days_to_expiry, errors="coerce")
    h = pd.to_numeric(horizon_days, errors="coerce")
    if pd.isna(p):
        return np.nan
    p = float(np.clip(float(p), 1e-6, 1.0 - 1e-6))
    if pd.isna(d) or pd.isna(h) or float(d) <= 0 or float(h) <= 0:
        return p
    try:
        hazard = -math.log(max(1.0 - p, 1e-6)) / max(float(d), 1.0)
        out = 1.0 - math.exp(-hazard * float(h))
        if not math.isfinite(out):
            return p
        return float(np.clip(out, 0.0, 1.0))
    except Exception:
        return p


def _normalize_text(parts: Iterable[object]) -> str:
    return " ".join(str(p).strip().lower() for p in parts if str(p).strip())


def _infer_theme_slot(row: pd.Series) -> str:
    code = str(row.get("basket_code", "")).strip().upper()
    text = _normalize_text(
        [
            row.get("title", ""),
            row.get("ticker_name", ""),
            row.get("event_slug", ""),
            row.get("event_family_key", ""),
            row.get("llm_category", ""),
        ]
    )
    if code == "ADIT-S3":
        if any(k in text for k in ["hormuz", "red sea", "shipping", "tanker", "pipeline", "lng", "oil supply"]):
            return "middle_east_shipping"
        if any(k in text for k in ["hezbollah", "lebanon", "beirut"]):
            return "middle_east_hezbollah"
        if any(k in text for k in ["hamas", "gaza", "rafah", "hostage", "ceasefire", "truce"]):
            return "middle_east_hamas"
        if "iran" in text and any(k in text for k in ["u.s.", "us ", "american", "america", "pentagon"]):
            return "middle_east_us_iran"
        if "iran" in text and "israel" in text:
            return "middle_east_iran_israel"
        return "middle_east_regional"
    if code == "ADIT-S4":
        if any(k in text for k in ["taiwan", "strait"]):
            return "us_china_taiwan"
        if any(k in text for k in ["tariff", "trade war", "export control", "chip", "semiconductor"]):
            return "us_china_trade"
        if any(k in text for k in ["south china sea", "navy", "military", "war", "strike"]):
            return "us_china_military"
    return str(row.get("event_family_key") or row.get("exclusive_group_key") or row.get("ticker_id") or row.get("market_id"))


def _resolve_payout(side: str, resolution_value: float | int | None) -> float:
    rv = pd.to_numeric(resolution_value, errors="coerce")
    if pd.isna(rv):
        return np.nan
    rv = float(rv)
    if side == "NO":
        return float(1.0 - rv)
    return rv


def _effective_trade_price(
    market_id: str,
    side: str,
    yes_prices_today: pd.Series,
    prev_instrument_prices: dict[str, float],
    instrument_id: str,
    *,
    allow_prev_for_existing: bool,
) -> float:
    eff_today = _effective_price_from_row(yes_prices_today.get(market_id, np.nan), side)
    if pd.notna(eff_today):
        return float(eff_today)
    if allow_prev_for_existing:
        eff_prev = prev_instrument_prices.get(instrument_id, np.nan)
        if pd.notna(eff_prev):
            return float(eff_prev)
    return np.nan


def _desired_weight_map_for_day(
    target_rows: pd.DataFrame,
    current_date: pd.Timestamp,
    yes_prices_today: pd.Series,
    yes_obs_dates_today: pd.Series,
    prev_instrument_prices: dict[str, float],
    holdings: dict[str, float],
    created_map: dict[str, pd.Timestamp],
    end_map: dict[str, pd.Timestamp],
    inactive_map: dict[str, pd.Timestamp],
    resolution_date_map: dict[str, pd.Timestamp],
    resolution_value_map: dict[str, float],
    cfg: ThematicNavConfig,
) -> tuple[dict[str, float], float, set[str], pd.DataFrame]:
    if target_rows.empty:
        return {}, 1.0, set(), pd.DataFrame()

    targets = target_rows[~target_rows["is_cash"]].copy()
    if targets.empty:
        return {}, 1.0, set(), pd.DataFrame()

    targets["market_id"] = targets["market_id"].astype(str)
    targets["position_side"] = targets["position_side"].fillna("YES").astype(str)
    targets["target_weight"] = pd.to_numeric(targets["target_weight"], errors="coerce").fillna(0.0)
    if "instrument_id" not in targets.columns:
        targets["instrument_id"] = targets["market_id"] + "::" + targets["position_side"]
    if "slot_key" not in targets.columns:
        targets["slot_key"] = targets.apply(_infer_theme_slot, axis=1)
    if "slot_name" not in targets.columns:
        targets["slot_name"] = targets["slot_key"]
    if "proxy_slot" not in targets.columns:
        targets["proxy_slot"] = False

    records: list[dict[str, object]] = []
    current_date = pd.Timestamp(current_date).normalize()
    for _, row in targets.iterrows():
        market_id = str(row["market_id"])
        side = str(row["position_side"])
        instrument_id = str(row["instrument_id"])
        position_instruction = "LONG_NO" if side == "NO" else "LONG_YES"
        listed_at = row.get("listed_at_nav", pd.NaT)
        if pd.isna(listed_at):
            listed_at = pd.to_datetime(row.get("listed_at"), errors="coerce")
        if pd.isna(listed_at):
            listed_at = created_map.get(market_id, pd.NaT)
        listed_at = pd.Timestamp(listed_at).normalize() if pd.notna(listed_at) else pd.NaT

        end_date = row.get("end_date_nav", pd.NaT)
        if pd.isna(end_date):
            end_date = pd.to_datetime(row.get("end_date"), errors="coerce")
        if pd.isna(end_date):
            end_date = end_map.get(market_id, pd.NaT)
        end_date = pd.Timestamp(end_date).normalize() if pd.notna(end_date) else pd.NaT

        inactive_date = row.get("inactive_at_nav", pd.NaT)
        if pd.isna(inactive_date):
            inactive_date = pd.to_datetime(row.get("inactive_at"), errors="coerce")
        if pd.isna(inactive_date):
            inactive_date = inactive_map.get(market_id, pd.NaT)
        inactive_date = pd.Timestamp(inactive_date).normalize() if pd.notna(inactive_date) else pd.NaT

        resolution_date = row.get("resolution_date_nav", pd.NaT)
        if pd.isna(resolution_date):
            resolution_date = resolution_date_map.get(market_id, pd.NaT)
        resolution_date = pd.Timestamp(resolution_date).normalize() if pd.notna(resolution_date) else pd.NaT
        resolution_value = pd.to_numeric(row.get("resolution_value_nav", resolution_value_map.get(market_id)), errors="coerce")
        roll_floor_days = pd.to_numeric(row.get("roll_floor_days"), errors="coerce")
        roll_floor_days = int(roll_floor_days) if pd.notna(roll_floor_days) else int(cfg.roll_dte_days)
        market_yes_price = pd.to_numeric(yes_prices_today.get(market_id, np.nan), errors="coerce")
        last_obs_date = yes_obs_dates_today.get(market_id, pd.NaT)
        last_obs_date = pd.Timestamp(last_obs_date).normalize() if pd.notna(last_obs_date) else pd.NaT
        price_staleness_days = (current_date - last_obs_date).days if pd.notna(last_obs_date) else np.nan
        effective_price_raw = _effective_price_from_row(market_yes_price, side)

        trade_price = _effective_trade_price(
            market_id,
            side,
            yes_prices_today,
            prev_instrument_prices,
            instrument_id,
            allow_prev_for_existing=instrument_id in holdings,
        )
        certainty = bool(pd.notna(trade_price) and (trade_price <= cfg.force_exit_price_low or trade_price >= cfg.force_exit_price_high))
        dte = (end_date - current_date).days if pd.notna(end_date) else np.nan
        roll_due = bool(
            pd.notna(dte)
            and int(roll_floor_days) > 0
            and int(dte) <= int(roll_floor_days)
        )
        settled = bool(
            cfg.settle_to_binary
            and pd.notna(resolution_date)
            and current_date >= resolution_date
            and pd.notna(resolution_value)
        )
        inactive = bool(pd.notna(inactive_date) and current_date >= inactive_date)
        listed = bool(pd.isna(listed_at) or current_date >= listed_at)
        investable = bool(
            listed
            and not inactive
            and not settled
            and not certainty
            and not roll_due
            and pd.notna(effective_price_raw)
            and float(effective_price_raw) > 0.0
        )
        records.append(
            {
                "instrument_id": instrument_id,
                "market_id": market_id,
                "basket_code": str(row.get("basket_code", "")),
                "slot_key": str(row["slot_key"]),
                "slot_name": str(row.get("slot_name", row["slot_key"])),
                "proxy_slot": bool(row.get("proxy_slot", False)),
                "position_side": side,
                "position_instruction": position_instruction,
                "market_yes_price": market_yes_price,
                "effective_price_raw": effective_price_raw,
                "effective_price_horizon": _horizon_normalized_probability(
                    effective_price_raw,
                    dte,
                    row.get("spot_horizon_days"),
                ),
                "days_to_expiry": dte,
                "price_staleness_days": price_staleness_days,
                "target_weight": float(row["target_weight"]),
                "direction_reason": str(row.get("direction_reason", "")),
                "preferred_probability_band": str(row.get("preferred_probability_band", "")),
                "tenor_band_status": str(row.get("tenor_band_status", "")),
                "roll_floor_days": roll_floor_days,
                "investable": investable,
                "excluded_reason": (
                    "not_listed" if not listed else
                    "inactive" if inactive else
                    "settled" if settled else
                    "certainty_exit" if certainty else
                    "roll_due" if roll_due else
                    "missing_price" if pd.isna(effective_price_raw) else
                    ""
                ),
            }
        )

    if not records:
        return {}, 1.0, set(), pd.DataFrame()
    rec = pd.DataFrame(records)
    total_non_cash = float(rec["target_weight"].sum())
    investable = rec[rec["investable"]].copy()
    coverage = float(investable["target_weight"].sum() / total_non_cash) if total_non_cash > 0 else 1.0
    if investable.empty:
        rec["desired_weight"] = 0.0
        rec["included_in_spot_series"] = False
        rec["excluded_reason"] = np.where(rec["excluded_reason"].astype(str).eq(""), "not_investable", rec["excluded_reason"])
        return {}, coverage, set(), rec

    invested_target = min(max(float(cfg.target_invested_weight), 1.0 - float(cfg.cash_cap)), 1.0 - float(cfg.cash_floor))
    if invested_target < 0:
        invested_target = 0.0

    desired: dict[str, float] = {}
    slot_budget = rec.groupby("slot_key", as_index=False)["target_weight"].sum().rename(columns={"target_weight": "slot_weight"})
    active_slots = set(investable["slot_key"].astype(str))
    slot_budget = slot_budget[slot_budget["slot_key"].astype(str).isin(active_slots)].copy()
    if slot_budget.empty:
        rec["desired_weight"] = 0.0
        rec["included_in_spot_series"] = False
        rec["excluded_reason"] = np.where(rec["excluded_reason"].astype(str).eq(""), "no_active_slots", rec["excluded_reason"])
        return {}, coverage, set(), rec

    if cfg.redistribute_unfilled_weight:
        slot_budget["slot_weight"] = slot_budget["slot_weight"] / max(float(slot_budget["slot_weight"].sum()), 1e-12)
        slot_budget["slot_weight"] = slot_budget["slot_weight"] * invested_target
    else:
        slot_budget["slot_weight"] = slot_budget["slot_weight"].clip(lower=0.0)

    slot_weight_map = slot_budget.set_index("slot_key")["slot_weight"].astype(float).to_dict()
    for slot_key, sg in investable.groupby("slot_key", sort=False):
        within_total = float(sg["target_weight"].sum())
        if within_total <= 0:
            continue
        slot_target = float(slot_weight_map.get(str(slot_key), 0.0))
        for _, row in sg.iterrows():
            desired[str(row["instrument_id"])] = slot_target * float(row["target_weight"]) / within_total

    rec["desired_weight"] = rec["instrument_id"].map(desired).fillna(0.0)
    rec["included_in_spot_series"] = (
        rec["desired_weight"] > 0
    ) & rec["effective_price_raw"].notna() & (pd.to_numeric(rec["price_staleness_days"], errors="coerce").fillna(9999) <= int(cfg.spot_stale_days))
    rec["excluded_reason"] = np.where(
        rec["included_in_spot_series"],
        "",
        np.where(
            rec["desired_weight"] <= 0,
            np.where(rec["excluded_reason"].astype(str).eq(""), "not_selected", rec["excluded_reason"]),
            np.where(
                pd.to_numeric(rec["price_staleness_days"], errors="coerce").fillna(9999) > int(cfg.spot_stale_days),
                "stale_price",
                np.where(rec["excluded_reason"].astype(str).eq(""), "spot_missing_price", rec["excluded_reason"]),
            ),
        ),
    )
    return desired, coverage, set(desired.keys()), rec


def build_daily_basket_nav(
    compositions: pd.DataFrame,
    processed_dir: str | Path,
    config: ThematicNavConfig | None = None,
    *,
    return_spot_diagnostics: bool = False,
    log_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config or ThematicNavConfig()
    if compositions.empty:
        empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        return empty if return_spot_diagnostics else empty[:2]

    c = compositions.copy()
    c["rebalance_date"] = pd.to_datetime(c["rebalance_date"], errors="coerce").dt.normalize()
    c["target_weight"] = pd.to_numeric(c["target_weight"], errors="coerce").fillna(0.0)
    c["basket_weight"] = pd.to_numeric(c["basket_weight"], errors="coerce").fillna(0.0)
    if "position_side" not in c.columns:
        c["position_side"] = "YES"
    c["position_side"] = c["position_side"].fillna("YES").astype(str)
    if "is_cash" not in c.columns:
        c["is_cash"] = False
    non_cash = c[~c["is_cash"]].copy()
    if non_cash.empty:
        empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        return empty if return_spot_diagnostics else empty[:2]

    market_ids = sorted(non_cash["market_id"].astype(str).unique())
    prices = load_selected_prices(processed_dir, market_ids)
    if prices.empty:
        empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        return empty if return_spot_diagnostics else empty[:2]
    lifecycle = load_market_lifecycle(processed_dir, market_ids)
    inactive_map = lifecycle.set_index("market_id")["inactive_date"].to_dict() if not lifecycle.empty else {}
    created_map = lifecycle.set_index("market_id")["created_date"].to_dict() if not lifecycle.empty else {}
    end_map = lifecycle.set_index("market_id")["end_date"].to_dict() if not lifecycle.empty else {}
    resolution_date_map = lifecycle.set_index("market_id")["resolution_date"].to_dict() if not lifecycle.empty else {}
    resolution_value_map = lifecycle.set_index("market_id")["resolution_value"].to_dict() if not lifecycle.empty else {}

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    raw_price_pivot = prices.pivot(index="date", columns="market_id", values="close_price").sort_index()
    obs_date_pivot = pd.DataFrame(index=raw_price_pivot.index, columns=raw_price_pivot.columns)
    for col in raw_price_pivot.columns:
        observed = pd.Series(raw_price_pivot.index.where(raw_price_pivot[col].notna()), index=raw_price_pivot.index)
        obs_date_pivot[col] = pd.to_datetime(observed, errors="coerce")

    basket_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    spot_diag_rows: list[dict] = []
    tc_rate = (cfg.fee_bps + cfg.spread_bps + cfg.slippage_bps) / 10000.0

    basket_groups = list(c.groupby("basket_code", sort=True))
    total_baskets = len(basket_groups)
    for basket_idx, (basket_code, grp) in enumerate(basket_groups, start=1):
        basket_all = grp.sort_values(["rebalance_date", "is_cash", "target_weight"], ascending=[True, True, False]).copy()
        basket_non_cash = basket_all[~basket_all["is_cash"]].copy()
        if basket_non_cash.empty:
            continue
        if log_progress:
            progress_pct = 100.0 * float(basket_idx - 1) / float(max(total_baskets, 1))
            print(
                f"[nav] {progress_pct:5.1f}% starting {basket_code} "
                f"({basket_idx}/{total_baskets})",
                flush=True,
            )

        basket_mids = sorted(basket_non_cash["market_id"].astype(str).unique())
        basket_roll_floor = pd.to_numeric(basket_non_cash.get("roll_floor_days"), errors="coerce")
        basket_roll_floor = int(basket_roll_floor.dropna().median()) if hasattr(basket_roll_floor, "dropna") and len(basket_roll_floor.dropna()) else int(cfg.roll_dte_days)
        px_raw = raw_price_pivot.reindex(columns=basket_mids)
        obs_raw = obs_date_pivot.reindex(columns=basket_mids)
        start_date = pd.Timestamp(basket_all["rebalance_date"].min()).normalize()
        end_candidates = [pd.Timestamp(basket_all["rebalance_date"].max()).normalize()]
        if not px_raw.empty and px_raw.index.notna().any():
            end_candidates.append(pd.Timestamp(px_raw.index.max()).normalize())
        end_date = max(end_candidates)
        calendar = pd.date_range(start_date, end_date, freq="D")
        px = px_raw.reindex(calendar).ffill()
        obs = obs_raw.reindex(calendar).ffill()

        targets_by_date: dict[pd.Timestamp, pd.DataFrame] = {
            pd.Timestamp(d).normalize(): dg.copy()
            for d, dg in basket_all.groupby("rebalance_date", sort=True)
        }
        rebalance_dates = sorted(targets_by_date)
        prepared_targets_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
        for rd, tg in targets_by_date.items():
            prep = tg.copy()
            prep["market_id"] = prep["market_id"].astype(str)
            prep["position_side"] = prep["position_side"].fillna("YES").astype(str)
            prep["target_weight"] = pd.to_numeric(prep["target_weight"], errors="coerce").fillna(0.0)
            prep["instrument_id"] = prep["market_id"] + "::" + prep["position_side"]
            prep["slot_key"] = prep.apply(_infer_theme_slot, axis=1)
            prep["listed_at_nav"] = pd.to_datetime(prep.get("listed_at"), errors="coerce").dt.normalize()
            prep["end_date_nav"] = pd.to_datetime(prep.get("end_date"), errors="coerce").dt.normalize()
            prep["inactive_at_nav"] = pd.to_datetime(prep.get("inactive_at"), errors="coerce").dt.normalize()
            prep["listed_at_nav"] = prep["listed_at_nav"].where(prep["listed_at_nav"].notna(), prep["market_id"].map(created_map))
            prep["end_date_nav"] = prep["end_date_nav"].where(prep["end_date_nav"].notna(), prep["market_id"].map(end_map))
            prep["inactive_at_nav"] = prep["inactive_at_nav"].where(prep["inactive_at_nav"].notna(), prep["market_id"].map(inactive_map))
            prep["resolution_date_nav"] = prep["market_id"].map(resolution_date_map)
            prep["resolution_value_nav"] = pd.to_numeric(prep["market_id"].map(resolution_value_map), errors="coerce")
            prepared_targets_by_date[rd] = prep

        holdings: dict[str, float] = {}
        avg_cost: dict[str, float] = {}
        cash = float(cfg.initial_capital)
        prev_nav = float(cfg.initial_capital)
        prev_instrument_prices: dict[str, float] = {}
        prev_date: pd.Timestamp | None = None
        target_price_coverage_share = 1.0
        current_target_date: pd.Timestamp | None = rebalance_dates[0] if rebalance_dates else None
        rebalance_idx = 0
        last_desired_keys: set[str] = set()
        spot_baseline: float | None = None

        total_days = len(calendar)
        for day_idx, d in enumerate(calendar, start=1):
            yes_prices_today = px.loc[d] if d in px.index else pd.Series(dtype=float)
            yes_obs_dates_today = obs.loc[d] if d in obs.index else pd.Series(dtype="datetime64[ns]")
            day_frac = 1.0 / 365.25 if prev_date is None else max((pd.Timestamp(d) - pd.Timestamp(prev_date)).days, 1) / 365.25
            cash *= (1.0 + float(cfg.annual_risk_free_rate)) ** day_frac

            entries = 0
            exits = 0
            day_turnover_notional = 0.0
            day_cost = 0.0
            rebalanced = False

            while rebalance_idx + 1 < len(rebalance_dates) and d >= rebalance_dates[rebalance_idx + 1]:
                rebalance_idx += 1
                current_target_date = rebalance_dates[rebalance_idx]
            scheduled_rebalance = bool(current_target_date is not None and d == current_target_date)

            # Expiry / resolution exits and certainty / pre-roll exits.
            for instrument_id in list(holdings.keys()):
                market_id, side = instrument_id.split("::", 1)
                eff_now = _effective_trade_price(
                    market_id,
                    side,
                    yes_prices_today,
                    prev_instrument_prices,
                    instrument_id,
                    allow_prev_for_existing=True,
                )
                inactive_date = inactive_map.get(market_id, pd.NaT)
                end_date = end_map.get(market_id, pd.NaT)
                resolution_date = resolution_date_map.get(market_id, pd.NaT)
                resolution_value = pd.to_numeric(resolution_value_map.get(market_id), errors="coerce")
                settled = bool(
                    cfg.settle_to_binary
                    and pd.notna(resolution_date)
                    and pd.Timestamp(d) >= pd.Timestamp(resolution_date)
                    and pd.notna(resolution_value)
                )
                expired = bool(pd.notna(inactive_date) and pd.Timestamp(d) >= pd.Timestamp(inactive_date))
                roll_due = bool(
                    pd.notna(end_date)
                    and pd.notna(basket_roll_floor)
                    and int(basket_roll_floor) > 0
                    and (pd.Timestamp(end_date).normalize() - pd.Timestamp(d).normalize()).days <= int(basket_roll_floor)
                )
                certainty = bool(pd.notna(eff_now) and (float(eff_now) <= cfg.force_exit_price_low or float(eff_now) >= cfg.force_exit_price_high))
                if not settled and not expired and not certainty and not roll_due:
                    continue
                if settled:
                    payout = _resolve_payout(side, resolution_value)
                    if pd.isna(payout):
                        continue
                    shares = holdings.pop(instrument_id)
                    avg_cost.pop(instrument_id, None)
                    prev_instrument_prices.pop(instrument_id, None)
                    cash += shares * float(payout)
                    exits += 1
                    continue
                if pd.isna(eff_now):
                    continue
                shares = holdings.pop(instrument_id)
                avg_cost.pop(instrument_id, np.nan)
                prev_instrument_prices.pop(instrument_id, None)
                notional = shares * float(eff_now)
                cash += notional
                exits += 1

            nav_pre_trade = cash
            for instrument_id, shares in holdings.items():
                market_id, side = instrument_id.split("::", 1)
                eff_trade = _effective_trade_price(
                    market_id,
                    side,
                    yes_prices_today,
                    prev_instrument_prices,
                    instrument_id,
                    allow_prev_for_existing=True,
                )
                if pd.notna(eff_trade):
                    nav_pre_trade += shares * float(eff_trade)

            target_rows = prepared_targets_by_date.get(current_target_date, pd.DataFrame()) if current_target_date is not None else pd.DataFrame()
            desired_weight_map, target_price_coverage_share, desired_keys, spot_diag = _desired_weight_map_for_day(
                target_rows,
                pd.Timestamp(d),
                yes_prices_today,
                yes_obs_dates_today,
                prev_instrument_prices,
                holdings,
                created_map,
                end_map,
                inactive_map,
                resolution_date_map,
                resolution_value_map,
                cfg,
            )
            current_cash_weight = float(cash / nav_pre_trade) if nav_pre_trade > 0 else 1.0
            needs_maintenance = bool(
                scheduled_rebalance
                or exits > 0
                or desired_keys != last_desired_keys
                or current_cash_weight > float(cfg.cash_cap) + 1e-9
                or any(instr not in desired_weight_map for instr in holdings)
            )

            if (cfg.immediate_redeploy or scheduled_rebalance) and needs_maintenance:
                rebalanced = True
                universe = sorted(set(holdings.keys()) | set(desired_weight_map.keys()))
                for instrument_id in universe:
                    market_id, side = instrument_id.split("::", 1)
                    trade_price = _effective_trade_price(
                        market_id,
                        side,
                        yes_prices_today,
                        prev_instrument_prices,
                        instrument_id,
                        allow_prev_for_existing=instrument_id in holdings,
                    )
                    if pd.isna(trade_price) or float(trade_price) <= 0:
                        continue
                    current_shares = holdings.get(instrument_id, 0.0)
                    current_notional = current_shares * float(trade_price)
                    target_notional = float(nav_pre_trade) * float(desired_weight_map.get(instrument_id, 0.0))
                    trade_notional = target_notional - current_notional
                    if abs(trade_notional) < 1e-10:
                        continue
                    trade_shares = trade_notional / float(trade_price)
                    trade_cost = abs(trade_notional) * tc_rate
                    cash -= trade_notional
                    cash -= trade_cost
                    day_cost += trade_cost
                    day_turnover_notional += abs(trade_notional)
                    new_shares = current_shares + trade_shares
                    if abs(new_shares) < 1e-10:
                        if instrument_id in holdings:
                            holdings.pop(instrument_id, None)
                            avg_cost.pop(instrument_id, None)
                            prev_instrument_prices.pop(instrument_id, None)
                            exits += 1
                    else:
                        holdings[instrument_id] = float(new_shares)
                        if trade_notional > 0 and current_shares <= 1e-10:
                            entries += 1
                        if trade_notional > 0:
                            old_cost = avg_cost.get(instrument_id, float(trade_price))
                            old_shares = max(current_shares, 0.0)
                            avg_cost[instrument_id] = float(
                                (old_cost * old_shares + float(trade_price) * trade_shares)
                                / max(old_shares + trade_shares, 1e-9)
                            )
                        elif instrument_id not in holdings:
                            avg_cost.pop(instrument_id, None)
                last_desired_keys = set(desired_weight_map.keys())

            invested = 0.0
            for instrument_id, shares in holdings.items():
                market_id, side = instrument_id.split("::", 1)
                eff_now = _effective_trade_price(
                    market_id,
                    side,
                    yes_prices_today,
                    prev_instrument_prices,
                    instrument_id,
                    allow_prev_for_existing=True,
                )
                if pd.notna(eff_now):
                    invested += shares * float(eff_now)
                    prev_instrument_prices[instrument_id] = float(eff_now)

            nav = float(cash + invested)
            turnover = float(day_turnover_notional / prev_nav / 2.0) if prev_nav > 0 else 0.0
            spot_price_coverage_share = 0.0
            spot_stale_weight_share = 0.0
            spot_weighted_effective_price_raw = np.nan
            spot_weighted_effective_price_horizon = np.nan
            spot_weighted_dte_days = np.nan
            spot_risk_level = np.nan
            slot_coverage_ratio = np.nan
            proxy_weight_share = np.nan
            tail_probability_weight_share = np.nan
            direction_balance_score = np.nan
            tenor_target_days = np.nan
            tenor_drift_days = np.nan

            if isinstance(spot_diag, pd.DataFrame) and not spot_diag.empty:
                diag = spot_diag.copy()
                diag["rebalance_date"] = pd.Timestamp(d).date().isoformat()
                diag["basket_code"] = str(basket_code)
                spot_diag_rows.extend(diag.to_dict("records"))

                desired_total = float(pd.to_numeric(diag["desired_weight"], errors="coerce").fillna(0.0).sum())
                included = diag.iloc[0:0].copy()
                if desired_total > 0:
                    included = diag[diag["included_in_spot_series"]].copy()
                    included_weight = float(pd.to_numeric(included["desired_weight"], errors="coerce").fillna(0.0).sum())
                    spot_price_coverage_share = included_weight / desired_total
                    stale_mask = pd.to_numeric(diag["price_staleness_days"], errors="coerce").fillna(9999) > int(cfg.spot_stale_days)
                    spot_stale_weight_share = float(pd.to_numeric(diag.loc[stale_mask, "desired_weight"], errors="coerce").fillna(0.0).sum()) / desired_total
                    proxy_weight_share = float(pd.to_numeric(diag.loc[diag["proxy_slot"].astype(bool), "desired_weight"], errors="coerce").fillna(0.0).sum()) / desired_total
                    tail_probability_weight_share = float(pd.to_numeric(diag.loc[diag["preferred_probability_band"].astype(str) == "tail", "desired_weight"], errors="coerce").fillna(0.0).sum()) / desired_total
                    yes_weight = float(pd.to_numeric(diag.loc[diag["position_instruction"] == "LONG_YES", "desired_weight"], errors="coerce").fillna(0.0).sum())
                    no_weight = float(pd.to_numeric(diag.loc[diag["position_instruction"] == "LONG_NO", "desired_weight"], errors="coerce").fillna(0.0).sum())
                    direction_balance_score = 1.0 - abs(yes_weight - no_weight) / desired_total
                    all_slots = {str(x) for x in diag["slot_key"].astype(str) if str(x)}
                    active_slots = {str(x) for x in diag.loc[diag["desired_weight"] > 0, "slot_key"].astype(str) if str(x)}
                    slot_coverage_ratio = float(len(active_slots) / len(all_slots)) if all_slots else 1.0
                if not included.empty:
                    included_weights = pd.to_numeric(included["desired_weight"], errors="coerce").fillna(0.0)
                    included_total = float(included_weights.sum())
                    if included_total > 0:
                        wn = included_weights / included_total
                        spot_weighted_effective_price_raw = float((pd.to_numeric(included["effective_price_raw"], errors="coerce").fillna(0.0) * wn).sum())
                        spot_weighted_effective_price_horizon = float((pd.to_numeric(included["effective_price_horizon"], errors="coerce").fillna(0.0) * wn).sum())
                        spot_weighted_dte_days = float((pd.to_numeric(included["days_to_expiry"], errors="coerce").fillna(0.0) * wn).sum())
                        target_vals_raw = (
                            target_rows["tenor_target_days"]
                            if (not target_rows.empty and "tenor_target_days" in target_rows.columns)
                            else pd.Series(dtype=float)
                        )
                        target_vals = pd.to_numeric(target_vals_raw, errors="coerce").dropna()
                        tenor_target_days = float(target_vals.median()) if not target_vals.empty else np.nan
                        if spot_baseline is None and spot_weighted_effective_price_horizon > 0:
                            spot_baseline = float(spot_weighted_effective_price_horizon)
                        if spot_baseline and spot_baseline > 0:
                            spot_risk_level = 100.0 * float(spot_weighted_effective_price_horizon) / float(spot_baseline)
                        if pd.notna(tenor_target_days):
                            tenor_drift_days = float(spot_weighted_dte_days - tenor_target_days)

            basket_rows.append(
                {
                    "rebalance_date": pd.Timestamp(d).date().isoformat(),
                    "domain": str(basket_all["domain"].iloc[0]),
                    "basket_code": str(basket_code),
                    "basket_name": str(basket_all["basket_name"].iloc[0]),
                    "basket_weight": float(basket_all["basket_weight"].iloc[0]),
                    "tradable_nav": nav,
                    "basket_level": nav,
                    "spot_risk_level": spot_risk_level,
                    "spot_weighted_effective_price_raw": spot_weighted_effective_price_raw,
                    "spot_weighted_effective_price_horizon": spot_weighted_effective_price_horizon,
                    "spot_weighted_dte_days": spot_weighted_dte_days,
                    "spot_price_coverage_share": spot_price_coverage_share,
                    "spot_stale_weight_share": spot_stale_weight_share,
                    "nav_spot_gap": (nav - spot_risk_level) if pd.notna(spot_risk_level) else np.nan,
                    "tenor_target_days": tenor_target_days,
                    "tenor_drift_days": tenor_drift_days,
                    "tail_probability_weight_share": tail_probability_weight_share,
                    "proxy_weight_share": proxy_weight_share,
                    "slot_coverage_ratio": slot_coverage_ratio,
                    "direction_balance_score": direction_balance_score,
                    "cash_weight": float(cash / nav) if nav > 0 else 1.0,
                    "price_coverage_share": float(target_price_coverage_share),
                    "entries": int(entries),
                    "exits": int(exits),
                    "turnover": turnover,
                    "transaction_cost": float(day_cost),
                    "rebalanced": bool(rebalanced),
                }
            )
            prev_nav = nav if nav > 0 else prev_nav
            prev_date = pd.Timestamp(d)
            if log_progress and (day_idx == total_days or day_idx == 1 or day_idx % 90 == 0):
                basket_progress = 100.0 * float(day_idx) / float(max(total_days, 1))
                print(
                    f"[nav] basket={basket_code} {basket_progress:5.1f}% "
                    f"day={pd.Timestamp(d).date().isoformat()} nav={nav:.2f}",
                    flush=True,
                )

    basket_df = pd.DataFrame(basket_rows)
    if basket_df.empty:
        triple = (basket_df, pd.DataFrame(), pd.DataFrame(spot_diag_rows))
        return triple if return_spot_diagnostics else triple[:2]

    basket_df["rebalance_date"] = pd.to_datetime(basket_df["rebalance_date"], errors="coerce")
    basket_df = basket_df.sort_values(["basket_code", "rebalance_date"]).reset_index(drop=True)

    agg_rows: list[dict] = []
    for d, g in basket_df.groupby("rebalance_date", sort=True):
        w = g["basket_weight"].astype(float)
        wsum = float(w.sum())
        wn = (w / wsum) if wsum > 0 else pd.Series(np.ones(len(g)) / max(len(g), 1), index=g.index)
        agg_rows.append(
            {
                "rebalance_date": pd.Timestamp(d).date().isoformat(),
                "overall_tradable_nav": float((g["tradable_nav"] * wn).sum()),
                "overall_basket_level": float((g["basket_level"] * wn).sum()),
                "overall_spot_risk_level": float((g["spot_risk_level"].fillna(0.0) * wn).sum()) if "spot_risk_level" in g.columns else np.nan,
                "overall_cash_weight": float((g["cash_weight"] * wn).sum()),
                "overall_price_coverage_share": float((g["price_coverage_share"] * wn).sum()),
                "overall_spot_price_coverage_share": float((g["spot_price_coverage_share"].fillna(0.0) * wn).sum()) if "spot_price_coverage_share" in g.columns else np.nan,
                "overall_nav_spot_gap": float((g["nav_spot_gap"].fillna(0.0) * wn).sum()) if "nav_spot_gap" in g.columns else np.nan,
                "n_baskets": int(g["basket_code"].nunique()),
                "total_entries": int(g["entries"].sum()),
                "total_exits": int(g["exits"].sum()),
                "total_turnover": float(g["turnover"].sum()),
                "rebalanced_baskets": int(g["rebalanced"].sum()),
            }
        )
    aggregate_df = pd.DataFrame(agg_rows).sort_values("rebalance_date").reset_index(drop=True)
    triple = (basket_df, aggregate_df, pd.DataFrame(spot_diag_rows))
    return triple if return_spot_diagnostics else triple[:2]
