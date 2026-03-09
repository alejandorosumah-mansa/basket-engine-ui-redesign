from __future__ import annotations

import pandas as pd

from src.prediction_basket.thematic_nav import ThematicNavConfig, build_daily_basket_nav


def _write_prices(tmp_path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(tmp_path / "prices.parquet", index=False)


def _write_history(tmp_path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(tmp_path / "polymarket_market_history.parquet", index=False)


def _base_compositions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rebalance_date": "2024-01-01",
                "domain": "CONFLICT",
                "basket_code": "ADIT-S3",
                "basket_name": "Middle East Armed Conflict",
                "basket_weight": 1.0,
                "market_id": "m_a",
                "ticker_id": "t_a",
                "ticker_name": "Israel strikes Iran?",
                "title": "Will Israel strike Iran?",
                "event_slug": "israel-strike-iran",
                "event_family_key": "israel-strike-iran",
                "exclusive_group_key": "israel-strike-iran",
                "llm_category": "middle_east",
                "position_side": "YES",
                "listed_at": "2023-12-01",
                "inactive_at": None,
                "end_date": "2024-12-31",
                "days_to_expiry": 365,
                "target_weight": 0.475,
                "is_cash": False,
            },
            {
                "rebalance_date": "2024-01-01",
                "domain": "CONFLICT",
                "basket_code": "ADIT-S3",
                "basket_name": "Middle East Armed Conflict",
                "basket_weight": 1.0,
                "market_id": "m_b",
                "ticker_id": "t_b",
                "ticker_name": "US and Iran conflict?",
                "title": "Will the U.S. strike Iran?",
                "event_slug": "us-strike-iran",
                "event_family_key": "us-strike-iran",
                "exclusive_group_key": "us-strike-iran",
                "llm_category": "middle_east",
                "position_side": "YES",
                "listed_at": "2023-12-01",
                "inactive_at": None,
                "end_date": "2024-12-31",
                "days_to_expiry": 365,
                "target_weight": 0.475,
                "is_cash": False,
            },
            {
                "rebalance_date": "2024-01-01",
                "domain": "CONFLICT",
                "basket_code": "ADIT-S3",
                "basket_name": "Middle East Armed Conflict",
                "basket_weight": 1.0,
                "market_id": "__CASH__",
                "ticker_id": "__CASH__",
                "ticker_name": "Cash Buffer",
                "title": "Cash Buffer",
                "event_slug": "__CASH__",
                "event_family_key": "__CASH__",
                "exclusive_group_key": "__CASH__",
                "llm_category": "cash",
                "position_side": "CASH",
                "listed_at": None,
                "inactive_at": None,
                "end_date": None,
                "days_to_expiry": None,
                "target_weight": 0.05,
                "is_cash": True,
            },
        ]
    )


def test_nav_immediately_redeploys_after_certainty_exit(tmp_path):
    comps = _base_compositions()
    _write_prices(
        tmp_path,
        [
            {"market_id": "m_a", "date": "2024-01-01", "close_price": 0.50, "volume": 1},
            {"market_id": "m_a", "date": "2024-01-02", "close_price": 0.96, "volume": 1},
            {"market_id": "m_b", "date": "2024-01-01", "close_price": 0.50, "volume": 1},
            {"market_id": "m_b", "date": "2024-01-02", "close_price": 0.50, "volume": 1},
        ],
    )
    _write_history(
        tmp_path,
        [
            {"market_id": "m_a", "created_at": "2023-12-01", "end_date": "2024-12-31", "closed_time": None, "resolution_time": None},
            {"market_id": "m_b", "created_at": "2023-12-01", "end_date": "2024-12-31", "closed_time": None, "resolution_time": None},
        ],
    )

    basket, _ = build_daily_basket_nav(comps, tmp_path, ThematicNavConfig())
    row = basket.loc[basket["rebalance_date"] == pd.Timestamp("2024-01-02")].iloc[0]

    assert row["rebalanced"]
    assert row["exits"] >= 1
    assert row["cash_weight"] <= 0.10 + 1e-9


def test_nav_pre_rolls_contracts_near_expiry(tmp_path):
    comps = _base_compositions()
    comps.loc[comps["market_id"] == "m_a", "end_date"] = "2024-01-10"
    _write_prices(
        tmp_path,
        [
            {"market_id": "m_a", "date": "2024-01-01", "close_price": 0.50, "volume": 1},
            {"market_id": "m_a", "date": "2024-01-02", "close_price": 0.50, "volume": 1},
            {"market_id": "m_b", "date": "2024-01-01", "close_price": 0.50, "volume": 1},
            {"market_id": "m_b", "date": "2024-01-02", "close_price": 0.80, "volume": 1},
        ],
    )
    _write_history(
        tmp_path,
        [
            {"market_id": "m_a", "created_at": "2023-12-01", "end_date": "2024-01-10", "closed_time": None, "resolution_time": None},
            {"market_id": "m_b", "created_at": "2023-12-01", "end_date": "2024-12-31", "closed_time": None, "resolution_time": None},
        ],
    )

    basket, _ = build_daily_basket_nav(comps, tmp_path, ThematicNavConfig(roll_dte_days=21))
    row = basket.loc[basket["rebalance_date"] == pd.Timestamp("2024-01-02")].iloc[0]

    assert row["basket_level"] > 150.0
    assert row["cash_weight"] <= 0.10 + 1e-9


def test_nav_uses_binary_settlement_when_resolution_value_exists(tmp_path):
    comps = _base_compositions()
    _write_prices(
        tmp_path,
        [
            {"market_id": "m_a", "date": "2024-01-01", "close_price": 0.50, "volume": 1},
            {"market_id": "m_a", "date": "2024-01-02", "close_price": 0.60, "volume": 1},
            {"market_id": "m_b", "date": "2024-01-01", "close_price": 0.50, "volume": 1},
            {"market_id": "m_b", "date": "2024-01-02", "close_price": 0.50, "volume": 1},
        ],
    )
    _write_history(
        tmp_path,
        [
            {
                "market_id": "m_a",
                "created_at": "2023-12-01",
                "end_date": "2024-12-31",
                "closed_time": "2024-01-02",
                "resolution_time": "2024-01-02",
                "resolution_value": 1.0,
            },
            {"market_id": "m_b", "created_at": "2023-12-01", "end_date": "2024-12-31", "closed_time": None, "resolution_time": None},
        ],
    )

    basket, _ = build_daily_basket_nav(comps, tmp_path, ThematicNavConfig(settle_to_binary=True))
    row = basket.loc[basket["rebalance_date"] == pd.Timestamp("2024-01-02")].iloc[0]

    assert row["basket_level"] > 140.0
    assert row["cash_weight"] <= 0.10 + 1e-9
