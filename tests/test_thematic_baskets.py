from __future__ import annotations

import pandas as pd

from src.prediction_basket.thematic_baskets import (
    _build_inception_policy,
    _build_website_html,
    default_specs,
    ThemePattern,
    ThematicBasketBuilder,
    ThematicBasketSpec,
)


def _sample_universe() -> pd.DataFrame:
    rows = []
    for i in range(1, 16):
        rows.append(
            {
                "market_id": f"m_{i:03d}",
                "ticker_id": f"t_{i:03d}",
                "ticker_name": f"AI contract {i}",
                "title": f"Will AI model release milestone {i} happen this year?",
                "title_l": f"will ai model release milestone {i} happen this year?",
                "end_date": pd.Timestamp("2026-12-31"),
                "quality_score": 0.50 + i * 0.01,
                "volume": 1000 + 10 * i,
                "platform": "polymarket",
                "current_price": 0.50,
                "listed_at": pd.Timestamp("2024-01-01"),
                "inactive_at": pd.NaT,
                "temporal_source": "test_fixture",
            }
        )
    rows.append(
        {
            "market_id": "m_election",
            "ticker_id": "t_election",
            "ticker_name": "Election market",
            "title": "Will election winner be announced soon?",
            "title_l": "will election winner be announced soon?",
            "end_date": pd.Timestamp("2026-12-31"),
            "quality_score": 0.99,
            "volume": 9999,
            "platform": "polymarket",
            "current_price": 0.50,
            "listed_at": pd.Timestamp("2024-01-01"),
            "inactive_at": pd.NaT,
            "temporal_source": "test_fixture",
        }
    )
    rows.append(
        {
            "market_id": "m_certainty",
            "ticker_id": "t_certainty",
            "ticker_name": "Certain AI contract",
            "title": "Will AI model ship this month?",
            "title_l": "will ai model ship this month?",
            "end_date": pd.Timestamp("2026-12-31"),
            "quality_score": 0.95,
            "volume": 9000,
            "platform": "polymarket",
            "current_price": 0.99,
            "listed_at": pd.Timestamp("2024-01-01"),
            "inactive_at": pd.NaT,
            "temporal_source": "test_fixture",
        }
    )
    return pd.DataFrame(rows)


def test_thematic_builder_enforces_cash_and_constraints():
    universe = _sample_universe()
    spec = ThematicBasketSpec(
        domain="AI_TECHNOLOGY",
        basket_code="ADIT-AI1",
        basket_name="AI Capability Advancement",
        basket_weight=1.0,
        include_patterns=(ThemePattern(r"\bai\b", 1.0),),
        direct_patterns=(ThemePattern(r"\bai\b|model release", 1.0),),
        allowed_categories=("ai_technology",),
        min_contracts=10,
        target_contracts=12,
        max_contracts=50,
        cash_weight=0.10,
    )

    builder = ThematicBasketBuilder(universe=universe, specs=[spec], min_days_to_expiry=21)
    out = builder.build_for_date("2026-03-01")

    assert not out.empty
    basket = out[out["basket_code"] == "ADIT-AI1"]
    non_cash = basket[~basket["is_cash"]]
    cash = basket[basket["is_cash"]]

    assert len(cash) == 1
    assert float(cash["target_weight"].iloc[0]) == 0.10
    assert 10 <= len(non_cash) <= 50
    assert non_cash["title"].str.lower().str.contains("election", regex=False).sum() == 0
    assert "m_certainty" not in set(non_cash["market_id"])

    total_w = float(basket["target_weight"].sum())
    assert abs(total_w - 1.0) < 1e-9


def test_strict_temporal_filter_excludes_future_listed_contracts():
    universe = _sample_universe()
    universe.loc[universe["market_id"] == "m_001", "listed_at"] = pd.Timestamp("2027-01-01")
    universe.loc[universe["market_id"] == "m_002", "listed_at"] = pd.NaT

    spec = ThematicBasketSpec(
        domain="AI_TECHNOLOGY",
        basket_code="ADIT-AI1",
        basket_name="AI Capability Advancement",
        basket_weight=1.0,
        include_patterns=(ThemePattern(r"\bai\b", 1.0),),
        direct_patterns=(ThemePattern(r"\bai\b|model release", 1.0),),
        allowed_categories=("ai_technology",),
        min_contracts=10,
        target_contracts=12,
        max_contracts=50,
        cash_weight=0.10,
    )
    builder = ThematicBasketBuilder(universe=universe, specs=[spec], min_days_to_expiry=21, strict_temporal=True)
    out = builder.build_for_date("2026-03-01")
    selected_ids = set(out.loc[~out["is_cash"], "market_id"].astype(str))

    assert "m_001" not in selected_ids
    assert "m_002" not in selected_ids


def test_inverse_pattern_uses_no_side_and_certainty_on_effective_price():
    universe = pd.DataFrame(
        [
            {
                "market_id": "m_strike",
                "ticker_id": "t_strike",
                "ticker_name": "Israel strike Iran?",
                "title": "Will Israel strike Iran before year end?",
                "title_l": "will israel strike iran before year end?",
                "end_date": pd.Timestamp("2026-12-31"),
                "quality_score": 0.8,
                "volume": 5000,
                "platform": "polymarket",
                "current_price": 0.62,
                "listed_at": pd.Timestamp("2024-01-01"),
                "inactive_at": pd.NaT,
                "temporal_source": "test_fixture",
                "event_slug": "israel-strike-iran-before-2026",
            },
            {
                "market_id": "m_cease_live",
                "ticker_id": "t_cease_live",
                "ticker_name": "Israel x Hamas ceasefire?",
                "title": "Will Israel x Hamas ceasefire before June?",
                "title_l": "will israel x hamas ceasefire before june?",
                "end_date": pd.Timestamp("2026-12-31"),
                "quality_score": 0.78,
                "volume": 4800,
                "platform": "polymarket",
                "current_price": 0.40,
                "listed_at": pd.Timestamp("2024-01-01"),
                "inactive_at": pd.NaT,
                "temporal_source": "test_fixture",
                "event_slug": "israel-x-hamas-ceasefire-before-june",
            },
            {
                "market_id": "m_cease_certain",
                "ticker_id": "t_cease_certain",
                "ticker_name": "Israel x Hamas ceasefire?",
                "title": "Will Israel x Hamas ceasefire before July?",
                "title_l": "will israel x hamas ceasefire before july?",
                "end_date": pd.Timestamp("2026-12-31"),
                "quality_score": 0.77,
                "volume": 4700,
                "platform": "polymarket",
                "current_price": 0.01,
                "listed_at": pd.Timestamp("2024-01-01"),
                "inactive_at": pd.NaT,
                "temporal_source": "test_fixture",
                "event_slug": "israel-x-hamas-ceasefire-before-july",
            },
        ]
    )

    spec = ThematicBasketSpec(
        domain="CONFLICT",
        basket_code="ADIT-S3",
        basket_name="Middle East Armed Conflict",
        basket_weight=1.0,
        include_patterns=(
            ThemePattern(r"israel|iran|hamas", 1.0),
            ThemePattern(r"strike|ceasefire|conflict|war", 1.0),
        ),
        direct_patterns=(ThemePattern(r"strike|attack|war|missile", 1.2),),
        inverse_patterns=(ThemePattern(r"ceasefire|truce|peace|deal", 1.2),),
        required_patterns=(r"israel|iran|hamas",),
        allowed_categories=("middle_east",),
        theme_polarity="risk_up",
        min_contracts=2,
        target_contracts=3,
        max_contracts=10,
        cash_weight=0.10,
        max_per_template=5,
        max_per_event_family=5,
    )

    builder = ThematicBasketBuilder(universe=universe, specs=[spec], min_days_to_expiry=21)
    out = builder.build_for_date("2026-03-01")
    non_cash = out[~out["is_cash"]].set_index("market_id")

    assert "m_strike" in non_cash.index
    assert "m_cease_live" in non_cash.index
    assert "m_cease_certain" not in non_cash.index
    assert non_cash.loc["m_strike", "position_side"] == "YES"
    assert non_cash.loc["m_cease_live", "position_side"] == "NO"
    assert abs(float(non_cash.loc["m_cease_live", "effective_price"]) - 0.60) < 1e-9


def test_llm_outcome_polarity_can_drive_no_side_for_risk_up_basket():
    universe = pd.DataFrame(
        [
            {
                "market_id": "m_accord",
                "ticker_id": "t_accord",
                "ticker_name": "Israel-Saudi accord?",
                "title": "Will Israel and Saudi Arabia sign an accord before July?",
                "title_l": "will israel and saudi arabia sign an accord before july?",
                "end_date": pd.Timestamp("2026-12-31"),
                "quality_score": 0.82,
                "volume": 5200,
                "platform": "polymarket",
                "current_price": 0.35,
                "listed_at": pd.Timestamp("2024-01-01"),
                "inactive_at": pd.NaT,
                "temporal_source": "test_fixture",
                "event_slug": "israel-saudi-accord-before-july",
                "llm_category": "middle_east",
                "classification_source": "test_fixture",
                "classification_confidence": 0.99,
                "yes_outcome_polarity": "risk_down",
                "yes_outcome_reason": "regional normalization reduces conflict risk",
                "exposure_confidence": 0.93,
            }
        ]
    )

    spec = ThematicBasketSpec(
        domain="CONFLICT",
        basket_code="ADIT-S3",
        basket_name="Middle East Armed Conflict",
        basket_weight=1.0,
        include_patterns=(
            ThemePattern(r"israel|saudi|arabia", 1.0),
            ThemePattern(r"accord|agreement|july", 0.5),
        ),
        direct_patterns=(ThemePattern(r"strike|attack|war|missile", 1.0),),
        inverse_patterns=(ThemePattern(r"ceasefire|truce", 1.0),),
        allowed_categories=("middle_east",),
        theme_polarity="risk_up",
        min_contracts=1,
        target_contracts=1,
        max_contracts=3,
        cash_weight=0.10,
        max_per_template=5,
        max_per_event_family=5,
    )

    builder = ThematicBasketBuilder(universe=universe, specs=[spec], min_days_to_expiry=21)
    out = builder.build_for_date("2026-03-01")
    non_cash = out[~out["is_cash"]].set_index("market_id")

    assert "m_accord" in non_cash.index
    assert non_cash.loc["m_accord", "position_side"] == "NO"
    assert str(non_cash.loc["m_accord", "side_source"]).startswith("llm_theme_risk_down")
    assert abs(float(non_cash.loc["m_accord", "effective_price"]) - 0.65) < 1e-9


def test_adit_s3_disables_tenor_pressure_and_disables_preroll():
    specs = {spec.basket_code: spec for spec in default_specs()}
    s3 = specs["ADIT-S3"]

    assert s3.dte_score_weight == 0.0
    assert s3.tenor_score_weight == 0.0
    assert s3.tenor_band_weight == 0.0
    assert s3.tenor_min_days == 0
    assert s3.tenor_max_days >= 3650
    assert s3.roll_floor_days == 0


def test_inception_policy_uses_auto_rule_and_manual_override():
    basket_level = pd.DataFrame(
        [
            {
                "rebalance_date": "2022-01-01",
                "basket_code": "ADIT-AI1",
                "basket_name": "AI Capability Advancement",
                "price_coverage_share": 0.20,
                "cash_weight": 1.00,
            },
            {
                "rebalance_date": "2023-03-01",
                "basket_code": "ADIT-AI1",
                "basket_name": "AI Capability Advancement",
                "price_coverage_share": 0.75,
                "cash_weight": 0.05,
            },
            {
                "rebalance_date": "2024-01-01",
                "basket_code": "ADIT-E3",
                "basket_name": "Oil Supply Disruption",
                "price_coverage_share": 0.20,
                "cash_weight": 0.90,
            },
            {
                "rebalance_date": "2024-02-01",
                "basket_code": "ADIT-E3",
                "basket_name": "Oil Supply Disruption",
                "price_coverage_share": 0.30,
                "cash_weight": 0.80,
            },
        ]
    )
    specs = [
        ThematicBasketSpec(
            domain="AI_TECHNOLOGY",
            basket_code="ADIT-AI1",
            basket_name="AI Capability Advancement",
            basket_weight=1.0,
            include_patterns=(ThemePattern(r"ai", 1.0),),
        ),
        ThematicBasketSpec(
            domain="ENERGY",
            basket_code="ADIT-E3",
            basket_name="Oil Supply Disruption",
            basket_weight=1.0,
            include_patterns=(ThemePattern(r"oil", 1.0),),
        ),
    ]
    overrides = {"ADIT-E3": {"date": "2024-02-01", "reason": "manual review"}}

    out = _build_inception_policy(basket_level, specs, overrides).set_index("basket_code")

    assert out.loc["ADIT-AI1", "default_inception_date"] == "2023-03-01"
    assert out.loc["ADIT-AI1", "effective_inception_date"] == "2023-03-01"
    assert out.loc["ADIT-AI1", "rule_name"] == "coverage>=0.60_cash<=0.25"
    assert out.loc["ADIT-E3", "default_inception_date"] == ""
    assert out.loc["ADIT-E3", "effective_inception_date"] == "2024-02-01"
    assert out.loc["ADIT-E3", "rule_name"] == "manual_override"
    assert out.loc["ADIT-E3", "override_reason"] == "manual review"


def test_website_builder_embeds_inception_payload_and_echarts(tmp_path):
    specs = [
        ThematicBasketSpec(
            domain="AI_TECHNOLOGY",
            basket_code="ADIT-AI1",
            basket_name="AI Capability Advancement",
            basket_weight=1.0,
            include_patterns=(ThemePattern(r"ai", 1.0),),
        )
    ]
    summary = pd.DataFrame(
        [
            {
                "domain": "AI_TECHNOLOGY",
                "basket_code": "ADIT-AI1",
                "basket_name": "AI Capability Advancement",
                "basket_weight": 1.0,
                "cash_weight": 0.05,
                "avg_contract_weight": 0.95,
                "avg_days_to_expiry": 120.0,
            }
        ]
    )
    monthly_summary = pd.DataFrame(
        [
            {
                "rebalance_date": "2023-03-01",
                "domain": "AI_TECHNOLOGY",
                "basket_code": "ADIT-AI1",
                "basket_name": "AI Capability Advancement",
                "basket_weight": 1.0,
                "n_contracts": 1,
                "avg_contract_weight": 0.95,
                "max_contract_weight": 0.95,
                "cash_weight": 0.05,
                "turnover": 0.10,
                "treasury_risk": 0.0,
                "broad_risk": 0.0,
            }
        ]
    )
    compositions = pd.DataFrame(
        [
            {
                "rebalance_date": "2023-03-01",
                "domain": "AI_TECHNOLOGY",
                "basket_code": "ADIT-AI1",
                "basket_name": "AI Capability Advancement",
                "basket_weight": 1.0,
                "market_id": "m_ai",
                "ticker_id": "t_ai",
                "title": "Will AI milestone happen?",
                "target_weight": 0.95,
                "end_date": "2023-12-31",
                "is_cash": False,
                "turnover": 0.10,
            },
            {
                "rebalance_date": "2023-03-01",
                "domain": "AI_TECHNOLOGY",
                "basket_code": "ADIT-AI1",
                "basket_name": "AI Capability Advancement",
                "basket_weight": 1.0,
                "market_id": "__CASH__",
                "ticker_id": "__CASH__",
                "title": "Cash",
                "target_weight": 0.05,
                "end_date": "",
                "is_cash": True,
                "turnover": 0.10,
            },
        ]
    )
    basket_level = pd.DataFrame(
        [
            {
                "rebalance_date": "2022-01-01",
                "domain": "AI_TECHNOLOGY",
                "basket_code": "ADIT-AI1",
                "basket_name": "AI Capability Advancement",
                "basket_level": 100.0,
                "cash_weight": 1.0,
                "price_coverage_share": 0.2,
                "entries": 0,
                "exits": 0,
                "turnover": 0.0,
                "rebalanced": True,
            },
            {
                "rebalance_date": "2023-03-01",
                "domain": "AI_TECHNOLOGY",
                "basket_code": "ADIT-AI1",
                "basket_name": "AI Capability Advancement",
                "basket_level": 112.0,
                "cash_weight": 0.05,
                "price_coverage_share": 0.8,
                "entries": 1,
                "exits": 0,
                "turnover": 0.1,
                "rebalanced": True,
            },
        ]
    )
    inception_policy = _build_inception_policy(basket_level, specs, {})

    _build_website_html(
        site_dir=tmp_path / "website",
        summary=summary,
        monthly_summary=monthly_summary,
        compositions=compositions,
        specs=specs,
        start="2022-01-01",
        end="2023-03-01",
        basket_level_series=basket_level,
        inception_policy=inception_policy,
    )

    index_html = (tmp_path / "website" / "index.html").read_text(encoding="utf-8")
    explorer_html = (tmp_path / "website" / "explorer.html").read_text(encoding="utf-8")

    assert "assets/echarts.min.js" in index_html
    assert "effective_inception_date" in index_html
    assert "Since Inception" in index_html
    assert "Full History" in index_html
    assert "echarts.min.js" in explorer_html
    assert (tmp_path / "website" / "assets" / "echarts.min.js").exists()
