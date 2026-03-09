"""Thematic basket construction with monthly rebalancing over market metadata.

This module builds explicit, rules-based thematic baskets from the contract
universe in ``data/processed/ticker_mapping.parquet``.

Design goals:
1. Strong theme specificity (AI, conflict, energy, governance, etc.).
2. Explicit exclusion of election-focused contracts.
3. Constrained basket sizes (10-50 contracts).
4. Monthly rebalance outputs with an explicit cash sleeve per basket.
5. Optional certainty filter hook when current price is available.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import hashlib
import json
import math
import html
import re
import shutil
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
import yaml

from src.prediction_basket.thematic_nav import attach_asof_prices, build_daily_basket_nav

INCEPTION_OVERRIDE_PATH = REPO_ROOT / "config" / "basket_inception_overrides.yml"
ECHARTS_ASSET_SOURCE = Path(__file__).resolve().parent / "assets" / "echarts.min.js"


def _get_plt():
    import os
    import tempfile

    mpl_config_dir = Path(tempfile.gettempdir()) / "basket_engine_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


GLOBAL_EXCLUDE_PATTERNS: tuple[str, ...] = (
    r"\belection\b",
    r"\bpresident\b",
    r"\bpresidential\b",
    r"\bsenate\b",
    r"\bsenator\b",
    r"\bhouse\b",
    r"\bgovernor\b",
    r"\bprimary\b",
    r"\bdemocrat\b",
    r"\brepublican\b",
    r"\bvote\b",
    r"\bballot\b",
    r"\bcampaign\b",
    r"\belectoral\b",
    r"\bgop\b",
    r"\bdnc\b",
    r"\brnc\b",
    r"\binauguration\b",
    # Broad sports/entertainment filters for cleaner macro/thematic baskets.
    r"\bnba\b",
    r"\bnfl\b",
    r"\bmlb\b",
    r"\bnhl\b",
    r"\buefa\b",
    r"\bfifa\b",
    r"\bworld cup\b",
    r"\bsuper bowl\b",
    r"\boscars?\b",
    r"\bgrammys?\b",
    r"\bbox office\b",
    r"\bbest picture\b",
    r"\bmovie\b",
    r"\btennis\b",
    r"\bgolf\b",
    r"\beurovision\b",
    r"\bgta vi\b",
    r"\bepstein\b",
    r"\btestif(?:y|ies|ied)\b",
    r"\bcontempt of congress\b",
    r"\bperson of the year\b",
    r"\bacademy awards?\b",
    r"\bdocumentary\b",
)


@dataclass(frozen=True)
class ThemePattern:
    pattern: str
    weight: float = 1.0


@dataclass(frozen=True)
class BasketSlotDefinition:
    slot_key: str
    slot_name: str
    include_patterns: tuple[ThemePattern, ...]
    inverse_patterns: tuple[ThemePattern, ...] = ()
    required_patterns: tuple[str, ...] = ()
    allowed_categories: tuple[str, ...] = ()
    min_names: int = 1
    max_names: int = 3
    required_if_available: bool = False
    proxy_slot: bool = False
    max_slot_weight: float = 0.25


@dataclass(frozen=True)
class ThematicBasketSpec:
    domain: str
    basket_code: str
    basket_name: str
    basket_weight: float
    include_patterns: tuple[ThemePattern, ...]
    direct_patterns: tuple[ThemePattern, ...] = ()
    inverse_patterns: tuple[ThemePattern, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    required_patterns: tuple[str, ...] = ()
    allowed_categories: tuple[str, ...] = ()
    theme_polarity: str = "risk_up"
    min_contracts: int = 10
    target_contracts: int = 20
    max_contracts: int = 50
    target_dte_days: int = 120
    cash_weight: float = 0.10
    max_single_weight: float = 0.12
    max_per_template: int = 50
    max_per_community: int = 999
    max_per_event_family: int = 1
    max_per_exclusive_group: int = 1
    dte_score_weight: float = 0.0
    tenor_target_days: int = 90
    tenor_min_days: int = 0
    tenor_max_days: int = 3650
    roll_floor_days: int = 0
    spot_horizon_days: int = 90
    tenor_score_weight: float = 0.0
    tenor_band_weight: float = 0.0
    preferred_effective_price_min: float = 0.15
    preferred_effective_price_max: float = 0.85
    soft_effective_price_min: float = 0.10
    soft_effective_price_max: float = 0.90
    proxy_weight_cap: float = 1.0
    slot_schema: tuple[BasketSlotDefinition, ...] = ()


def _compile_union(patterns: Iterable[str]) -> re.Pattern[str] | None:
    p = [x for x in patterns if x]
    if not p:
        return None
    return re.compile("|".join(f"(?:{x})" for x in p), flags=re.IGNORECASE)


def _zscore(s: pd.Series) -> pd.Series:
    std = float(s.std(ddof=0))
    if not math.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - float(s.mean())) / std


def _rank_pct(s: pd.Series) -> pd.Series:
    if s.empty:
        return s.astype(float)
    return s.rank(pct=True, method="average").fillna(0.0)


def _score_text_patterns(texts: pd.Series, patterns: Iterable[ThemePattern]) -> pd.Series:
    score = pd.Series(np.zeros(len(texts)), index=texts.index, dtype=float)
    if score.empty:
        return score
    txt = texts.fillna("").astype(str)
    for p in patterns:
        mask = txt.str.contains(p.pattern, regex=True, na=False)
        score.loc[mask] = score.loc[mask] + float(p.weight)
    return score


def _first_existing_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([None] * len(df), index=df.index)


def _rowwise_min_datetime(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = pd.to_datetime(a, errors="coerce", utc=True)
    bb = pd.to_datetime(b, errors="coerce", utc=True)
    return pd.concat([aa, bb], axis=1).min(axis=1)


def _rowwise_max_datetime(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = pd.to_datetime(a, errors="coerce", utc=True)
    bb = pd.to_datetime(b, errors="coerce", utc=True)
    return pd.concat([aa, bb], axis=1).max(axis=1)


_TEMPLATE_STOPWORDS = {
    "will", "the", "a", "an", "by", "before", "after", "in", "on", "of", "to",
    "at", "be", "is", "are", "there", "have", "has", "with", "and", "or",
    "from", "this", "that", "next", "new", "least", "most", "more", "less",
    "than", "year", "month", "day", "end", "start", "meeting", "between",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}

_ENTITY_WORDS = {
    "openai", "anthropic", "xai", "deepseek", "gemini", "google", "meta", "microsoft",
    "mistral", "alibaba", "meituan", "claude", "grok",
    "tesla", "trump", "starmer", "john", "oliver", "xi", "jinping", "putin", "zelenskyy",
    "israel", "iran", "hamas", "hezbollah", "houthi", "russia", "ukraine", "china", "taiwan",
}

_EVENT_TIME_TOKENS = {
    "by", "before", "after", "in", "on", "of", "end", "start", "next", "this",
    "jan", "january", "feb", "february", "mar", "march", "apr", "april",
    "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept",
    "september", "oct", "october", "nov", "november", "dec", "december",
    "q1", "q2", "q3", "q4", "year", "month", "week", "day",
}

_EVENT_OUTCOME_TOKENS = {
    "yes", "no", "change", "increase", "decrease", "up", "down", "higher", "lower",
    "over", "under", "greater", "less", "between", "exactly", "least", "most",
    "high", "low", "hit", "settle", "settles", "rate", "rates",
}

_CATEGORY_FALLBACK = {"unknown", "other", ""}

_THEME_CATEGORY_ALIASES = {
    "ai_technology": "ai_technology",
    "china_us": "china_geopolitics",
    "china_geopolitics": "china_geopolitics",
    "middle_east": "middle_east",
    "russia_ukraine": "russia_ukraine",
    "energy_commodities": "energy_commodities",
    "pandemic_health": "pandemic_health",
    "fed_monetary_policy": "fed_monetary_policy",
    "us_economic": "us_economic",
    "legal_regulatory": "legal_regulatory",
    "space_frontier": "space_frontier",
    "europe_politics": "global_politics",
    "climate_environment": "global_politics",
    "sports_entertainment": "sports_entertainment",
    "pop_culture_misc": "other",
    "crypto_digital": "other",
    "us_elections": "us_elections",
    "global_politics": "global_politics",
}

_HEURISTIC_CATEGORY_PATTERNS: dict[str, tuple[ThemePattern, ...]] = {
    "ai_technology": (
        ThemePattern(r"\bai\b|artificial intelligence|\bllm\b|language model|chatgpt|openai|anthropic|deepseek|xai|grok", 1.4),
        ThemePattern(r"model release|frontier model|best ai model|agi|autonomous|robot", 1.1),
        ThemePattern(r"gpu|nvidia|compute|datacenter|inference", 0.9),
    ),
    "china_geopolitics": (
        ThemePattern(r"\bchina\b|taiwan|beijing|\bpla\b|tsmc|south china sea|rare earth", 1.4),
        ThemePattern(r"\bxi\b|jinping|ccp|communist party|politburo", 1.0),
    ),
    "middle_east": (
        ThemePattern(r"israel|iran|gaza|hamas|hezbollah|houthi|tehran|hormuz|red sea", 1.5),
        ThemePattern(r"saudi|uae|lebanon|syria|iraq|yemen|idf", 1.0),
    ),
    "russia_ukraine": (
        ThemePattern(r"ukraine|russia|putin|kremlin|moscow|kyiv", 1.4),
        ThemePattern(r"nato|baltic|poland|crimea|donbas", 1.0),
    ),
    "energy_commodities": (
        ThemePattern(r"\boil\b|crude|brent|\bwti\b|opec|lng|natural gas|pipeline|refinery|tanker", 1.4),
        ThemePattern(r"hormuz|shipping|production|supply disruption|facility", 1.0),
    ),
    "pandemic_health": (
        ThemePattern(r"pandemic|outbreak|virus|covid|h5n1|ebola|mpox|measles|bird flu", 1.5),
        ThemePattern(r"vaccine|\bcdc\b|\bfda\b|world health organization|\bwho\b|health emergency", 1.0),
    ),
    "fed_monetary_policy": (
        ThemePattern(r"\bfed\b|fomc|federal reserve|interest rates?|rate cut|rate hike|fed chair", 1.5),
        ThemePattern(r"ecb|boe|boj|central bank", 0.8),
    ),
    "us_economic": (
        ThemePattern(r"recession|gdp|unemployment|inflation|\bcpi\b|\bppi\b|housing|retail sales|payrolls", 1.4),
        ThemePattern(r"soft landing|economy|economic|yield curve|default|bankrupt", 1.0),
    ),
    "legal_regulatory": (
        ThemePattern(r"congress|supreme court|shutdown|debt ceiling|doj|\bsec\b|\bftc\b|regulation|policy|rule", 1.3),
        ThemePattern(r"ban|vacancy|ruling|lawsuit|tariff", 0.9),
    ),
    "space_frontier": (
        ThemePattern(r"spacex|starship|rocket|launch|nasa|satellite|orbit|orbital", 1.4),
        ThemePattern(r"moon|mars|lunar|space station", 1.0),
    ),
    "global_politics": (
        ThemePattern(r"war|conflict|sanction|ceasefire|military|summit|diplomatic|treaty|sovereign", 1.1),
        ThemePattern(r"government|cabinet|prime minister|parliament|trade war", 0.8),
    ),
}


_RANK_OUTCOME_RE = re.compile(
    r"(?:\bbest\b|\btop\b|\bfirst\b|\bsecond\b|\bthird\b|\bfourth\b|#\s*[1-4]|\b[1-4](?:st|nd|rd|th)?\b|number\s*[1-4])",
    flags=re.IGNORECASE,
)


def _tokenize_title(text: str) -> list[str]:
    t = str(text).lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)
    toks = [w for w in t.split() if len(w) >= 2]
    return toks


def _template_key_from_title(text: str) -> str:
    toks = _tokenize_title(text)
    cleaned = [
        w for w in toks
        if (w not in _TEMPLATE_STOPWORDS and w not in _ENTITY_WORDS and not w.isdigit())
    ]
    if not cleaned:
        cleaned = toks[:6]
    cleaned = cleaned[:10]
    if not cleaned:
        return "generic-template"
    return " ".join(cleaned)


def _jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    if inter == 0:
        return 0.0
    union = len(tokens_a | tokens_b)
    if union == 0:
        return 0.0
    return inter / union


def _normalize_builder_category(value: object) -> str:
    raw = str(value or "").strip().lower().replace(" ", "_")
    if not raw or raw == "nan":
        return "unknown"
    return str(_THEME_CATEGORY_ALIASES.get(raw, raw))


def _build_search_text(frame: pd.DataFrame) -> pd.Series:
    title = frame.get("title_l", frame.get("title", "")).fillna("").astype(str)
    ticker_name = frame.get("ticker_name", "").fillna("").astype(str).str.lower()
    event_slug = frame.get("event_slug", "").fillna("").astype(str).str.lower().str.replace("-", " ", regex=False)
    search = (title + " " + ticker_name + " " + event_slug).str.replace(r"\s+", " ", regex=True).str.strip()
    return search


def _apply_semantic_heuristics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "title" not in out.columns:
        out["title"] = ""
    out["title"] = out["title"].fillna("").astype(str)
    out["title_l"] = out.get("title_l", out["title"].str.lower()).fillna("").astype(str)
    if "ticker_name" not in out.columns:
        out["ticker_name"] = ""
    if "event_slug" not in out.columns:
        out["event_slug"] = ""
    if "llm_category" not in out.columns:
        out["llm_category"] = "unknown"
    if "llm_secondary_category" not in out.columns:
        out["llm_secondary_category"] = ""
    if "classification_source" not in out.columns:
        out["classification_source"] = "unknown"
    if "classification_confidence" not in out.columns:
        out["classification_confidence"] = 0.0
    if "exposure_direction" not in out.columns:
        out["exposure_direction"] = ""
    if "exposure_description" not in out.columns:
        out["exposure_description"] = ""
    if "exposure_confidence" not in out.columns:
        out["exposure_confidence"] = 0.0
    if "yes_outcome_polarity" not in out.columns:
        out["yes_outcome_polarity"] = ""
    if "yes_outcome_reason" not in out.columns:
        out["yes_outcome_reason"] = ""
    if "direction_model" not in out.columns:
        out["direction_model"] = ""

    out["llm_category"] = out["llm_category"].map(_normalize_builder_category).fillna("unknown")
    out["llm_secondary_category"] = out["llm_secondary_category"].map(_normalize_builder_category).fillna("")
    out["classification_source"] = out["classification_source"].fillna("unknown").astype(str)
    out["classification_confidence"] = pd.to_numeric(out["classification_confidence"], errors="coerce").fillna(0.0)
    out["exposure_direction"] = out["exposure_direction"].fillna("").astype(str).str.lower()
    out["exposure_description"] = out["exposure_description"].fillna("").astype(str)
    out["exposure_confidence"] = pd.to_numeric(out["exposure_confidence"], errors="coerce").fillna(0.0)
    out["yes_outcome_polarity"] = out["yes_outcome_polarity"].fillna("").astype(str).str.lower()
    out["yes_outcome_reason"] = out["yes_outcome_reason"].fillna("").astype(str)
    out["direction_model"] = out["direction_model"].fillna("").astype(str)
    out["search_text"] = _build_search_text(out)

    unknown_mask = out["llm_category"].isin(_CATEGORY_FALLBACK)
    if unknown_mask.any():
        txt = out.loc[unknown_mask, "search_text"]
        cat_scores = {
            cat: _score_text_patterns(txt, patterns)
            for cat, patterns in _HEURISTIC_CATEGORY_PATTERNS.items()
        }
        if cat_scores:
            score_df = pd.DataFrame(cat_scores, index=txt.index)
            best_score = score_df.max(axis=1)
            best_cat = score_df.idxmax(axis=1)
            assign_mask = best_score >= 1.0
            if assign_mask.any():
                assign_idx = best_score.index[assign_mask]
                out.loc[assign_idx, "llm_category"] = best_cat.loc[assign_idx].astype(str)
                out.loc[assign_idx, "classification_source"] = "heuristic_category"
                out.loc[assign_idx, "classification_confidence"] = best_score.loc[assign_idx].clip(lower=0.0, upper=3.0) / 3.0

    return out


def _event_family_key(event_slug: str, title: str) -> str:
    slug = str(event_slug or "").strip().lower()
    if not slug or slug == "nan":
        return _template_key_from_title(title)

    toks = [t for t in slug.split("-") if t]
    cleaned: list[str] = []
    for tok in toks:
        if tok in _EVENT_TIME_TOKENS or tok in _EVENT_OUTCOME_TOKENS:
            continue
        if re.fullmatch(r"\d{1,4}", tok):
            continue
        if re.fullmatch(r"20\d\d", tok):
            continue
        cleaned.append(tok)

    if not cleaned:
        return _template_key_from_title(title)
    return "-".join(cleaned[:12])


def _mutual_exclusion_key(event_slug: str, title: str) -> str:
    slug = str(event_slug or "").strip().lower()
    ttl = str(title or "").strip().lower()
    if not slug and not ttl:
        return ""

    text = f"{slug} {ttl}".strip()
    # Target explicit leaderboard/ranking markets (main pain-point in AI baskets).
    ai_hint = ("ai-model" in slug) or ("ai model" in ttl)
    comp_hint = ("which-compan" in slug) or ("which company" in ttl) or (slug.startswith("will-") and "-have-" in slug)
    if not (ai_hint and comp_hint and _RANK_OUTCOME_RE.search(text)):
        return ""

    base = slug if slug and slug != "nan" else ttl
    base = re.sub(r"-\d{2,}$", "", base)
    base = base.replace("style-control-on", "")
    base = re.sub(r"^will-[a-z0-9-]+-have-", "will-entity-have-", base)
    base = re.sub(r"\b(?:best|top|first|second|third|fourth|1st|2nd|3rd|4th)\b", "rank", base)
    base = re.sub(r"#\s*[1-4]", "rank", base)
    base = re.sub(r"\b[1-4]\b", "rank", base)
    base = re.sub(r"[^a-z0-9]+", "-", base)
    base = re.sub(r"-+", "-", base).strip("-")
    if not base:
        return "mx-ai-ranking"

    toks = [t for t in base.split("-") if t]
    toks = ["entity" if t in _ENTITY_WORDS else t for t in toks]
    collapsed: list[str] = []
    for t in toks:
        if collapsed and collapsed[-1] == t:
            continue
        collapsed.append(t)
    if not collapsed:
        return "mx-ai-ranking"
    return "mx-ai-ranking-" + "-".join(collapsed[:20])


def _clip_and_normalize(weights: pd.Series, max_weight: float) -> pd.Series:
    if weights.empty:
        return weights

    w = weights.clip(lower=0).astype(float)
    if float(w.sum()) <= 0:
        w[:] = 1.0
    w = w / float(w.sum())

    for _ in range(12):
        capped = w.clip(upper=max_weight)
        excess = float((w - capped).clip(lower=0).sum())
        w = capped
        if excess <= 1e-12:
            break
        free = w[w < max_weight - 1e-12]
        if free.empty:
            break
        w.loc[free.index] = w.loc[free.index] + excess * (w.loc[free.index] / float(free.sum()))

    if float(w.sum()) <= 0:
        w[:] = 1.0
    return w / float(w.sum())


def _make_position_instruction(side: object) -> str:
    sval = str(side or "").strip().upper()
    if sval == "NO":
        return "LONG_NO"
    if sval == "YES":
        return "LONG_YES"
    return "CASH"


def _direction_reason(
    search_text: object,
    position_side: object,
    side_source: object,
    direct_score: object,
    inverse_score: object,
) -> str:
    text = str(search_text or "").lower()
    side = str(position_side or "").upper()
    source = str(side_source or "").lower()
    dscore = float(pd.to_numeric(direct_score, errors="coerce") or 0.0)
    iscore = float(pd.to_numeric(inverse_score, errors="coerce") or 0.0)
    if side == "CASH":
        return "cash_buffer"
    if side == "NO":
        if source.startswith("llm_theme_"):
            return f"{source}_held_on_no"
        if re.search(r"ceasefire|truce|peace|deal|normalize|recognition|resume shipping|de.?escalat", text):
            return "de-escalation_market_held_on_no"
        if "exposure" in source:
            return "exposure_cache_held_on_no"
        if iscore >= dscore:
            return "inverse_signal_held_on_no"
        return "long_no"
    if source.startswith("llm_theme_"):
        return f"{source}_held_on_yes"
    if re.search(r"strike|attack|war|missile|raid|inva|military|clash|close the strait|supply disrupt|shipping disrupt|opec cut|production cut", text):
        return "risk_up_market_held_on_yes"
    if "exposure" in source:
        return "exposure_cache_held_on_yes"
    return "long_yes"


def _classify_probability_band(
    effective_price: object,
    *,
    certainty_low: float,
    certainty_high: float,
    preferred_min: float,
    preferred_max: float,
    soft_min: float,
    soft_max: float,
) -> str:
    p = pd.to_numeric(effective_price, errors="coerce")
    if pd.isna(p):
        return "unknown"
    p = float(p)
    if p <= certainty_low or p >= certainty_high:
        return "certain"
    if preferred_min <= p <= preferred_max:
        return "preferred"
    if soft_min <= p <= soft_max:
        return "soft"
    return "tail"


def _probability_band_bonus(band: str) -> float:
    return {
        "preferred": 1.0,
        "soft": 0.45,
        "tail": -0.85,
        "certain": -1.75,
        "unknown": 0.0,
    }.get(str(band or "unknown"), 0.0)


def _tenor_band_bonus(status: str) -> float:
    return {
        "in_band": 0.65,
        "below_band": -0.35,
        "above_band": -0.45,
        "fallback": -0.10,
    }.get(str(status or "fallback"), -0.10)


def _tenor_band_status(days_to_expiry: object, spec: ThematicBasketSpec) -> str:
    dte = pd.to_numeric(days_to_expiry, errors="coerce")
    if pd.isna(dte):
        return "fallback"
    dte = float(dte)
    if dte < float(spec.tenor_min_days):
        return "below_band"
    if dte > float(spec.tenor_max_days):
        return "above_band"
    return "in_band"


def _slot_allowed_category(
    llm_category: object,
    llm_secondary_category: object,
    slot: BasketSlotDefinition,
) -> bool:
    if not slot.allowed_categories:
        return True
    allowed = {str(x) for x in slot.allowed_categories}
    primary = str(llm_category or "")
    secondary = str(llm_secondary_category or "")
    return primary in allowed or secondary in allowed or primary in _CATEGORY_FALLBACK


def _slot_match_info(frame: pd.DataFrame, slot: BasketSlotDefinition) -> pd.DataFrame:
    text = frame["search_text"].fillna("").astype(str)
    score = _score_text_patterns(text, slot.include_patterns) + _score_text_patterns(text, slot.inverse_patterns)
    required_ok = pd.Series(True, index=frame.index)
    if slot.required_patterns:
        req = _compile_union(slot.required_patterns)
        if req is not None:
            required_ok = text.str.contains(req, na=False)
    category_ok = pd.Series(
        [
            _slot_allowed_category(
                frame.at[idx, "llm_category"] if "llm_category" in frame.columns else "",
                frame.at[idx, "llm_secondary_category"] if "llm_secondary_category" in frame.columns else "",
                slot,
            )
            for idx in frame.index
        ],
        index=frame.index,
    )
    out = pd.DataFrame(
        {
            "slot_score": score,
            "slot_required_ok": required_ok,
            "slot_category_ok": category_ok,
        },
        index=frame.index,
    )
    out["slot_match"] = out["slot_required_ok"] & out["slot_category_ok"] & (out["slot_score"] > 0)
    return out


def _apply_slot_caps(weights: pd.Series, slot_keys: pd.Series, slot_caps: dict[str, float]) -> pd.Series:
    w = weights.astype(float).copy()
    if w.empty:
        return w
    if not slot_caps:
        total = float(w.sum())
        return w / total if total > 0 else w

    slot_keys = slot_keys.reindex(w.index).fillna("")
    for _ in range(24):
        total = float(w.sum())
        if total <= 0:
            break
        w = w / total
        changed = False
        excess_total = 0.0
        violating_slots: set[str] = set()
        for slot_key, cap in slot_caps.items():
            if cap <= 0:
                continue
            idx = slot_keys[slot_keys.astype(str) == str(slot_key)].index
            if len(idx) == 0:
                continue
            slot_sum = float(w.reindex(idx).sum())
            if slot_sum > cap + 1e-9:
                scale = cap / max(slot_sum, 1e-12)
                before = w.reindex(idx).copy()
                w.loc[idx] = w.loc[idx] * scale
                excess_total += float(before.sum() - w.loc[idx].sum())
                violating_slots.add(str(slot_key))
                changed = True
        if not changed or excess_total <= 1e-12:
            break
        free_idx = slot_keys[~slot_keys.astype(str).isin(violating_slots)].index
        if len(free_idx) == 0:
            break
        free_weights = w.reindex(free_idx).clip(lower=0.0)
        if float(free_weights.sum()) <= 0:
            w.loc[free_idx] = excess_total / len(free_idx)
        else:
            w.loc[free_idx] = w.loc[free_idx] + excess_total * (free_weights / float(free_weights.sum()))
    total = float(w.sum())
    return w / total if total > 0 else w


def _horizon_normalized_probability(effective_price: object, days_to_expiry: object, horizon_days: object) -> float:
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


def _slot(
    slot_key: str,
    slot_name: str,
    include_patterns: tuple[ThemePattern, ...],
    *,
    inverse_patterns: tuple[ThemePattern, ...] = (),
    required_patterns: tuple[str, ...] = (),
    allowed_categories: tuple[str, ...] = (),
    min_names: int = 1,
    max_names: int = 3,
    required_if_available: bool = False,
    proxy_slot: bool = False,
    max_slot_weight: float = 0.25,
) -> BasketSlotDefinition:
    return BasketSlotDefinition(
        slot_key=slot_key,
        slot_name=slot_name,
        include_patterns=include_patterns,
        inverse_patterns=inverse_patterns,
        required_patterns=required_patterns,
        allowed_categories=allowed_categories,
        min_names=min_names,
        max_names=max_names,
        required_if_available=required_if_available,
        proxy_slot=proxy_slot,
        max_slot_weight=max_slot_weight,
    )


FAST_RISK_DEFAULTS = {
    "dte_score_weight": 0.0,
    "tenor_target_days": 90,
    "tenor_min_days": 0,
    "tenor_max_days": 3650,
    "roll_floor_days": 0,
    "spot_horizon_days": 90,
    "tenor_score_weight": 0.0,
    "tenor_band_weight": 0.0,
    "preferred_effective_price_min": 0.15,
    "preferred_effective_price_max": 0.85,
    "soft_effective_price_min": 0.10,
    "soft_effective_price_max": 0.90,
}


STRUCTURAL_DEFAULTS = {
    "dte_score_weight": 0.0,
    "tenor_target_days": 180,
    "tenor_min_days": 0,
    "tenor_max_days": 3650,
    "roll_floor_days": 0,
    "spot_horizon_days": 180,
    "tenor_score_weight": 0.0,
    "tenor_band_weight": 0.0,
    "preferred_effective_price_min": 0.15,
    "preferred_effective_price_max": 0.85,
    "soft_effective_price_min": 0.10,
    "soft_effective_price_max": 0.90,
}


def _annotate_slot_fields(frame: pd.DataFrame, spec: ThematicBasketSpec) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        for col, value in [
            ("slot_key", ""),
            ("slot_name", ""),
            ("proxy_slot", False),
            ("slot_score", 0.0),
            ("slot_required_if_available", False),
            ("slot_max_weight", np.nan),
        ]:
            out[col] = value
        return out

    if not spec.slot_schema:
        out["slot_key"] = ""
        out["slot_name"] = ""
        out["proxy_slot"] = False
        out["slot_score"] = 0.0
        out["slot_required_if_available"] = False
        out["slot_max_weight"] = np.nan
        return out

    best_score = pd.Series(np.full(len(out), -np.inf), index=out.index, dtype=float)
    best_key = pd.Series(["unassigned"] * len(out), index=out.index, dtype=object)
    best_name = pd.Series(["Unassigned"] * len(out), index=out.index, dtype=object)
    best_proxy = pd.Series([False] * len(out), index=out.index, dtype=bool)
    best_required = pd.Series([False] * len(out), index=out.index, dtype=bool)
    best_cap = pd.Series([np.nan] * len(out), index=out.index, dtype=float)

    for slot in spec.slot_schema:
        info = _slot_match_info(out, slot)
        score = info["slot_score"].where(info["slot_match"], -np.inf)
        better = score > best_score
        if better.any():
            best_score.loc[better] = score.loc[better]
            best_key.loc[better] = slot.slot_key
            best_name.loc[better] = slot.slot_name
            best_proxy.loc[better] = slot.proxy_slot
            best_required.loc[better] = slot.required_if_available
            best_cap.loc[better] = float(slot.max_slot_weight)

    out["slot_key"] = best_key.astype(str)
    out["slot_name"] = best_name.astype(str)
    out["proxy_slot"] = best_proxy.astype(bool)
    out["slot_score"] = pd.Series(best_score).replace([-np.inf], 0.0).astype(float)
    out["slot_required_if_available"] = best_required.astype(bool)
    out["slot_max_weight"] = best_cap.astype(float)
    return out


def default_specs() -> list[ThematicBasketSpec]:
    specs = [
        ThematicBasketSpec(
            domain="AI_TECHNOLOGY",
            basket_code="ADIT-AI1",
            basket_name="AI Capability Advancement",
            basket_weight=0.2814,
            include_patterns=(
                ThemePattern(r"\bai\b", 1.0),
                ThemePattern(r"artificial intelligence", 1.2),
                ThemePattern(r"openai|anthropic|gemini|chatgpt|xai|deepseek", 1.1),
                ThemePattern(r"\bllm\b|language model|model release", 0.9),
                ThemePattern(r"gpu|nvidia|compute|datacenter", 0.7),
                ThemePattern(r"robot|autonomous|\bagi\b", 0.7),
            ),
            direct_patterns=(
                ThemePattern(r"\bai\b|artificial intelligence|\bllm\b|language model", 1.0),
                ThemePattern(r"frontier model|model release|model launch|agi|autonomous|robot", 1.1),
                ThemePattern(r"gpu|nvidia|compute|datacenter|inference", 0.8),
            ),
            inverse_patterns=(
                ThemePattern(r"ban|moratorium|pause training|regulatory block|model shutdown|delay|fail(?:ure|s)?", 1.1),
            ),
            theme_polarity="growth_up",
            required_patterns=(r"\bai\b|artificial intelligence|model|llm|gpu|compute|autonomous|robot",),
            allowed_categories=("ai_technology",),
            exclude_patterns=(
                r"\bsay\b",
                r"person of the year",
                r"last week tonight",
                r"prime ministers questions",
                r"state of the union",
                r"\btoken\b|\bfdv\b|billboard|song",
                r"\bsecond best\b|\bthird best\b|\bfourth best\b|second-best|third-best|fourth-best",
                r"\bsecond\b.*\bai model\b|\bthird\b.*\bai model\b|\bfourth\b.*\bai model\b",
            ),
            min_contracts=15,
            target_contracts=18,
            max_single_weight=0.10,
            max_per_template=1,
            max_per_community=3,
            **STRUCTURAL_DEFAULTS,
        ),
        ThematicBasketSpec(
            domain="CONFLICT",
            basket_code="ADIT-S4",
            basket_name="US-China Conflict Escalation",
            basket_weight=0.2232,
            include_patterns=(
                ThemePattern(r"us.?china|china.?us", 1.2),
                ThemePattern(r"taiwan|south china sea|taiwan strait", 1.0),
                ThemePattern(r"trade war|tariff|sanction", 0.9),
                ThemePattern(r"beijing|\bpla\b|rare earth|tsmc", 0.7),
            ),
            direct_patterns=(
                ThemePattern(r"invad|blockade|clash|strike|military|war|sanction|tariff|export control", 1.2),
                ThemePattern(r"taiwan strait|south china sea|rare earth|chip ban|tsmc", 0.9),
            ),
            inverse_patterns=(
                ThemePattern(r"deal|truce|peace|summit|meeting|reconciliation|detente", 1.0),
            ),
            required_patterns=(
                r"(?=.*(?:china|taiwan|beijing|\bpla\b|tsmc|south china sea|rare earth))(?=.*(?:invade|blockade|clash|strike|military|war|tariff|sanction|trade|strait|chip))",
            ),
            allowed_categories=("china_geopolitics", "global_politics", "us_military", "legal_regulatory", "us_economic"),
            exclude_patterns=(r"state of the union|say \"?taiwan\"?", r"gta vi"),
            min_contracts=15,
            target_contracts=18,
            max_per_template=2,
            max_per_community=4,
            max_per_event_family=2,
            **FAST_RISK_DEFAULTS,
        ),
        ThematicBasketSpec(
            domain="CONFLICT",
            basket_code="ADIT-SP1",
            basket_name="Space Infrastructure Expansion",
            basket_weight=0.2343,
            include_patterns=(
                ThemePattern(r"\bspace\b|orbital|orbit", 1.1),
                ThemePattern(r"spacex|starship|nasa", 1.1),
                ThemePattern(r"satellite|launch|rocket", 1.0),
                ThemePattern(r"\bmoon\b|\bmars\b|lunar", 0.8),
            ),
            direct_patterns=(
                ThemePattern(r"launch|rocket|starship|satellite|orbit|orbital|moon|mars|lunar", 1.1),
            ),
            inverse_patterns=(
                ThemePattern(r"explod|fail(?:ure|s)?|scrub|grounded|cancel", 1.1),
            ),
            theme_polarity="growth_up",
            required_patterns=(r"space|spacex|starship|nasa|satellite|launch|rocket|orbit|lunar|mars",),
            allowed_categories=("space_frontier",),
            exclude_patterns=(r"\btoken\b|\bfdv\b|metamask|crypto",),
            min_contracts=15,
            target_contracts=16,
            max_per_template=1,
            max_per_community=5,
            **STRUCTURAL_DEFAULTS,
        ),
        ThematicBasketSpec(
            domain="CONFLICT",
            basket_code="ADIT-S3",
            basket_name="Middle East Armed Conflict",
            basket_weight=0.2843,
            include_patterns=(
                ThemePattern(r"middle east|iran|israel|gaza", 1.2),
                ThemePattern(r"hamas|hezbollah|houthi|red sea", 1.0),
                ThemePattern(r"saudi|uae|lebanon|syria|iraq|yemen|tehran", 0.9),
            ),
            direct_patterns=(
                ThemePattern(r"strike|attack|war|missile|rocket|raid|inva|target|military|escalat|conflict|close the strait", 1.3),
                ThemePattern(r"red sea|hormuz|hezbollah|houthi", 0.9),
            ),
            inverse_patterns=(
                ThemePattern(r"ceasefire|truce|peace|deal|normaliz|hostage release|de.?escalat|recognition", 1.4),
            ),
            required_patterns=(
                r"(?=.*(?:iran|israel|gaza|hamas|hezbollah|houthi|saudi|tehran|lebanon|syria|iraq|yemen))(?=.*(?:strike|ceasefire|attack|conflict|war|missile|normaliz|intervention|military|inva|target))",
            ),
            allowed_categories=("middle_east", "global_politics", "us_military", "energy_commodities"),
            exclude_patterns=(r"fed chair|trump nominate|nobel|peace prize",),
            min_contracts=15,
            target_contracts=18,
            max_per_template=1,
            max_per_community=5,
            slot_schema=(
                _slot(
                    "me_iran_israel",
                    "Iran-Israel Direct Escalation",
                    (
                        ThemePattern(r"iran.*israel|israel.*iran", 1.6),
                        ThemePattern(r"strike|attack|missile|raid|war|inva|target", 1.2),
                    ),
                    allowed_categories=("middle_east", "global_politics"),
                    required_if_available=True,
                    max_slot_weight=0.24,
                ),
                _slot(
                    "me_hamas_ceasefire_failure",
                    "Gaza / Hamas Ceasefire Failure",
                    (
                        ThemePattern(r"hamas|gaza|hostage", 1.4),
                        ThemePattern(r"ceasefire|truce|deal|phase ii|phase 2|phase two", 1.3),
                    ),
                    inverse_patterns=(ThemePattern(r"ceasefire|truce|deal|phase ii|phase 2|phase two", 1.4),),
                    allowed_categories=("middle_east",),
                    required_if_available=True,
                    max_slot_weight=0.22,
                ),
                _slot(
                    "me_hezbollah_lebanon",
                    "Hezbollah / Lebanon Front",
                    (
                        ThemePattern(r"hezbollah|lebanon|beirut", 1.5),
                        ThemePattern(r"strike|attack|war|missile|front", 1.0),
                    ),
                    allowed_categories=("middle_east",),
                    required_if_available=True,
                    max_slot_weight=0.18,
                ),
                _slot(
                    "me_red_sea_hormuz",
                    "Red Sea / Hormuz / Shipping Disruption",
                    (
                        ThemePattern(r"red sea|hormuz|shipping|tanker|maersk|houthi|strait", 1.5),
                        ThemePattern(r"close|attack|disrupt|hit|block", 1.0),
                    ),
                    inverse_patterns=(ThemePattern(r"resume shipping|shipping normalize|reopen", 1.3),),
                    allowed_categories=("middle_east", "energy_commodities"),
                    required_if_available=True,
                    max_slot_weight=0.18,
                ),
                _slot(
                    "me_us_iran",
                    "US-Iran Intervention Risk",
                    (
                        ThemePattern(r"u\\.s\\.|\\bus\\b|american|pentagon", 1.0),
                        ThemePattern(r"iran|tehran", 1.3),
                        ThemePattern(r"strike|attack|intervention|war|conflict", 1.2),
                    ),
                    allowed_categories=("middle_east", "global_politics"),
                    required_if_available=True,
                    max_slot_weight=0.18,
                ),
                _slot(
                    "me_normalization_failure",
                    "Regional Normalization Failure",
                    (
                        ThemePattern(r"normalize|normalisation|recognition|diplomatic|deal", 1.2),
                        ThemePattern(r"saudi|uae|indonesia|syria|lebanon|israel", 1.0),
                    ),
                    inverse_patterns=(ThemePattern(r"normalize|normalisation|recognition|deal|resume shipping|de.?escalat", 1.4),),
                    allowed_categories=("middle_east", "global_politics"),
                    required_if_available=True,
                    max_slot_weight=0.15,
                ),
            ),
            **FAST_RISK_DEFAULTS,
        ),
        ThematicBasketSpec(
            domain="CONFLICT",
            basket_code="ADIT-S2",
            basket_name="Europe War & NATO Conflict",
            basket_weight=0.1252,
            include_patterns=(
        ThemePattern(r"nato|ukraine|russia|putin|kremlin", 1.2),
        ThemePattern(r"european war|europe conflict|baltic|poland", 0.9),
        ThemePattern(r"europe|eu troops|germany|france|united kingdom|\buk\b|greenland|denmark|arctic|nordic|finland|sweden", 0.6),
    ),
    direct_patterns=(
        ThemePattern(r"war|conflict|strike|capture|clash|troops|offensive|mobiliz|nato|acquire|tariff|dispute|tension|security fracture", 1.2),
    ),
    inverse_patterns=(
        ThemePattern(r"ceasefire|peace|truce|deal|settlement", 1.2),
    ),
    required_patterns=(
        r"(?=.*(?:nato|ukraine|russia|putin|kremlin|baltic|poland|greenland|denmark|arctic|nordic|finland|sweden))(?=.*(?:war|conflict|strike|capture|ceasefire|clash|troops|security guarantee|acquire|tariff|dispute|tension))",
    ),
            allowed_categories=("russia_ukraine", "global_politics", "us_military"),
            exclude_patterns=(r"nobel|eurovision|gta vi|maduro",),
            min_contracts=15,
            target_contracts=16,
            max_per_template=1,
            max_per_community=4,
            slot_schema=(
                _slot(
                    "eu_russia_territorial",
                    "Russia Territorial Advance",
                    (
                        ThemePattern(r"ukraine|russia|donetsk|crimea|kharkiv|zaporizhzhia", 1.4),
                        ThemePattern(r"capture|advance|offensive|take|occupy|territorial", 1.2),
                    ),
                    allowed_categories=("russia_ukraine", "global_politics"),
                    required_if_available=True,
                    max_slot_weight=0.22,
                ),
                _slot(
                    "eu_ceasefire_failure",
                    "Ukraine Ceasefire Failure",
                    (
                        ThemePattern(r"ukraine|russia|ceasefire|peace|truce|deal", 1.4),
                    ),
                    inverse_patterns=(ThemePattern(r"ceasefire|peace|truce|deal|security guarantee", 1.4),),
                    allowed_categories=("russia_ukraine", "global_politics"),
                    required_if_available=True,
                    max_slot_weight=0.18,
                ),
                _slot(
                    "eu_nato_direct",
                    "NATO Direct Clash / Member Strike",
                    (
                        ThemePattern(r"nato|poland|baltic|member", 1.4),
                        ThemePattern(r"strike|clash|military clash|attack|article 5", 1.2),
                    ),
                    allowed_categories=("russia_ukraine", "global_politics"),
                    required_if_available=True,
                    max_slot_weight=0.18,
                ),
                _slot(
                    "eu_greenland_arctic",
                    "Greenland / Arctic / Nordic Tension",
                    (
                        ThemePattern(r"greenland|denmark|arctic|nordic|finland|sweden|baltic", 1.5),
                        ThemePattern(r"acquire|tariff|dispute|security|clash|tension", 1.0),
                    ),
                    allowed_categories=("global_politics", "russia_ukraine"),
                    required_if_available=True,
                    max_slot_weight=0.16,
                ),
                _slot(
                    "eu_nuclear_strategic",
                    "Nuclear / Strategic Escalation",
                    (
                        ThemePattern(r"nuclear|strategic|missile|tactical", 1.4),
                        ThemePattern(r"russia|ukraine|nato|putin", 1.0),
                    ),
                    allowed_categories=("russia_ukraine", "global_politics"),
                    required_if_available=True,
                    max_slot_weight=0.14,
                ),
                _slot(
                    "eu_security_fracture",
                    "EU / NATO Security Fracture",
                    (
                        ThemePattern(r"security guarantee|nato secretary general|eu troops|france|germany|uk", 1.1),
                        ThemePattern(r"fracture|break|withdraw|security|coalition", 1.0),
                    ),
                    inverse_patterns=(ThemePattern(r"deal|peace|security guarantee", 1.2),),
                    allowed_categories=("global_politics", "russia_ukraine"),
                    required_if_available=True,
                    max_slot_weight=0.12,
                ),
            ),
            **FAST_RISK_DEFAULTS,
        ),
        ThematicBasketSpec(
            domain="ENERGY",
            basket_code="ADIT-E3",
            basket_name="Oil Supply Disruption",
            basket_weight=0.2753,
            include_patterns=(
                ThemePattern(r"\boil\b|\bcrude\b|\bbrent\b|\bwti\b", 1.2),
                ThemePattern(r"opec|pipeline|hormuz", 1.0),
                ThemePattern(r"natural gas|lng|energy crisis", 0.8),
            ),
            direct_patterns=(
                ThemePattern(r"above|over|hit high|settle above|supply disrupt|shipping disrupt|tanker|pipeline attack|facility hit|terminal hit|refinery hit|hormuz", 1.2),
                ThemePattern(r"opec cut|production cut|oil spike|crude spike|shortfall|output miss", 1.0),
            ),
            inverse_patterns=(
                ThemePattern(r"below|under|hit low|settle below|production rise|production increase|supply increase|shipping normalize|resume shipping|ceasefire|supply restored|reach .*barrels per day|at least .*barrels per day", 1.1),
            ),
            required_patterns=(
                r"(?=.*(?:oil|crude|brent|wti|hormuz|opec|pipeline|lng|natural gas))(?=.*(?:price|production|supply|disrupt|shipping|tanker|facility|settle|hit))",
            ),
            allowed_categories=("energy_commodities", "middle_east", "global_politics"),
            exclude_patterns=(r"academy awards?|camera|documentary|\btoken\b|\bfdv\b",),
            min_contracts=15,
            target_contracts=16,
            max_per_template=2,
            max_per_community=4,
            max_per_event_family=3,
            proxy_weight_cap=0.35,
            slot_schema=(
                _slot(
                    "oil_hormuz_redsea",
                    "Hormuz / Red Sea Disruption",
                    (
                        ThemePattern(r"hormuz|red sea|houthi|strait", 1.6),
                        ThemePattern(r"close|disrupt|attack|shipping|tanker|block", 1.1),
                    ),
                    inverse_patterns=(ThemePattern(r"resume shipping|shipping normalize|reopen", 1.3),),
                    allowed_categories=("energy_commodities", "middle_east"),
                    required_if_available=True,
                    max_slot_weight=0.22,
                ),
                _slot(
                    "oil_infra_hits",
                    "Export / Terminal / Refinery Hits",
                    (
                        ThemePattern(r"terminal|refinery|kharg|pipeline|export facility|oil facility", 1.5),
                        ThemePattern(r"hit|attack|fire|explosion|disrupt", 1.0),
                    ),
                    required_patterns=(r"(terminal|refinery|kharg|pipeline|export facility|oil facility)",),
                    allowed_categories=("energy_commodities", "middle_east"),
                    required_if_available=True,
                    max_slot_weight=0.20,
                ),
                _slot(
                    "oil_opec_supply",
                    "OPEC / Production Restriction",
                    (
                        ThemePattern(r"opec|production cut|output cut|supply cut", 1.5),
                    ),
                    inverse_patterns=(ThemePattern(r"production increase|supply increase|opec hike", 1.3),),
                    allowed_categories=("energy_commodities",),
                    required_if_available=True,
                    max_slot_weight=0.18,
                ),
                _slot(
                    "oil_shipping_disruption",
                    "Shipping / Tanker Disruption",
                    (
                        ThemePattern(r"shipping|tanker|maersk|freight|route|vessel", 1.3),
                        ThemePattern(r"disrupt|attack|delay|reroute|block", 1.0),
                    ),
                    inverse_patterns=(ThemePattern(r"resume shipping|shipping normalize", 1.2),),
                    allowed_categories=("energy_commodities", "middle_east"),
                    required_if_available=True,
                    max_slot_weight=0.16,
                ),
                _slot(
                    "oil_price_proxy",
                    "Oil Price Proxy",
                    (
                        ThemePattern(r"oil over|oil above|crude over|crude above|brent over|wti over|hit high|settle above", 1.4),
                        ThemePattern(r"oil|crude|brent|wti", 0.9),
                    ),
                    inverse_patterns=(ThemePattern(r"oil under|crude under|brent under|wti under|below|hit low|settle below|\blow\b", 1.2),),
                    allowed_categories=("energy_commodities",),
                    required_if_available=True,
                    proxy_slot=True,
                    max_slot_weight=0.35,
                ),
                _slot(
                    "oil_producer_shortfall",
                    "Producer Output Shortfall",
                    (
                        ThemePattern(r"venezuela|saudi|iraq|iran|russia|producer", 1.1),
                        ThemePattern(r"shortfall|below|under|production miss|output miss|supply disruption|barrels per day|\bbpd\b", 1.1),
                    ),
                    inverse_patterns=(ThemePattern(r"production increase|supply restored|reach .*barrels per day|at least .*barrels per day", 1.1),),
                    required_patterns=(r"(production|output|barrels per day|bpd|supply)",),
                    allowed_categories=("energy_commodities", "middle_east"),
                    required_if_available=True,
                    max_slot_weight=0.14,
                ),
            ),
            **FAST_RISK_DEFAULTS,
        ),
        ThematicBasketSpec(
            domain="GENERAL",
            basket_code="ADIT-X4",
            basket_name="Pandemic & Health Crisis",
            basket_weight=0.2818,
            include_patterns=(
                ThemePattern(r"\bpandemic\b|outbreak|health emergency", 1.2),
                ThemePattern(r"covid|virus|h5n1|ebola|mpox|measles|\bflu\b", 1.0),
                ThemePattern(r"\bcdc\b|world health organization|\bfda\b|vaccine", 0.7),
            ),
            direct_patterns=(
                ThemePattern(r"pandemic|outbreak|spread|health emergency|new variant|cases rise", 1.3),
                ThemePattern(r"covid|virus|h5n1|ebola|mpox|measles|bird flu", 1.0),
            ),
            inverse_patterns=(
                ThemePattern(r"vaccine approval|contained|eradicated|no outbreak|avoid outbreak", 1.0),
            ),
            required_patterns=(
                r"(?=.*(?:pandemic|outbreak|covid|virus|h5n1|ebola|mpox|measles|cdc|fda|vaccine))(?=.*(?:new|cases?|variant|warning|approval|emergency|spread))",
            ),
            allowed_categories=("pandemic_health",),
            min_contracts=15,
            target_contracts=16,
            max_per_template=4,
            max_per_community=4,
            max_per_event_family=4,
            **FAST_RISK_DEFAULTS,
        ),
        ThematicBasketSpec(
            domain="GENERAL",
            basket_code="ADIT-X3",
            basket_name="Major Economy Recession",
            basket_weight=0.1634,
            include_patterns=(
                ThemePattern(r"recession|gdp|unemployment", 1.1),
                ThemePattern(r"inflation|cpi|ppi|housing|retail sales", 1.0),
                ThemePattern(r"\bfed\b|rates?|yield|treasury|default|bankrupt", 0.8),
            ),
            direct_patterns=(
                ThemePattern(r"recession|gdp contraction|unemployment rise|default|bankrupt|bank failure", 1.3),
                ThemePattern(r"inflation|cpi|ppi|yield curve|treasury|hard landing", 1.0),
            ),
            inverse_patterns=(
                ThemePattern(r"soft landing|avoid recession|no recession|disinflation", 1.2),
            ),
            required_patterns=(
                r"(?=.*(?:recession|gdp|unemployment|inflation|cpi|ppi|fed|yield|treasury|default|bankrupt|rate))(?=.*(?:us|u\.s|eurozone|china|japan|germany|canada|uk|economy|economic|federal reserve|ecb|boe|central bank))",
            ),
            allowed_categories=("us_economic", "fed_monetary_policy", "global_politics", "legal_regulatory"),
            exclude_patterns=(r"trump nominate|fed chair\?|military clash",),
            min_contracts=15,
            target_contracts=18,
            max_per_template=1,
            max_per_community=5,
            **FAST_RISK_DEFAULTS,
        ),
        ThematicBasketSpec(
            domain="POLITICAL",
            basket_code="ADIT-P1",
            basket_name="China Governance",
            basket_weight=0.0956,
            include_patterns=(
                ThemePattern(r"\bxi\b|ccp|communist party", 1.2),
                ThemePattern(r"china governance|china leadership|china policy", 1.0),
                ThemePattern(r"beijing|china regulator", 0.7),
            ),
            direct_patterns=(
                ThemePattern(r"out|removed|resign|purge|succession|policy shock|crackdown|politburo", 1.2),
            ),
            inverse_patterns=(
                ThemePattern(r"remain|stay in power|continuity|meet with|visit", 1.0),
            ),
            required_patterns=(
                r"(?=.*(?:china|ccp|communist party|xi|beijing|politburo|premier|state council))(?=.*(?:leadership|governance|policy|regulator|general secretary|party congress|out|removed|resign|term|succession|cabinet|purge))",
            ),
            allowed_categories=("china_geopolitics", "global_politics"),
            exclude_patterns=(r"divorce|nobel|peace prize|meet with|visit us",),
            min_contracts=15,
            target_contracts=16,
            max_per_template=2,
            max_per_community=4,
            max_per_event_family=2,
            **FAST_RISK_DEFAULTS,
        ),
        ThematicBasketSpec(
            domain="POLITICAL",
            basket_code="ADIT-P4",
            basket_name="US Governance",
            basket_weight=0.1973,
            include_patterns=(
                ThemePattern(r"congress|supreme court|white house", 1.1),
                ThemePattern(r"budget|shutdown|debt ceiling", 1.0),
                ThemePattern(r"fed chair|doj|\bsec\b|\bftc\b|regulation|policy", 0.8),
            ),
            direct_patterns=(
                ThemePattern(r"shutdown|debt ceiling|ban|vacancy|regulation|policy|ruling|court|rule", 1.2),
            ),
            inverse_patterns=(
                ThemePattern(r"avoid shutdown|deal reached|budget passed", 1.0),
            ),
            required_patterns=(
                r"(?=.*(?:us|u\.s|united states|congress|supreme court|white house|federal|doj|sec|ftc|fed chair))(?=.*(?:budget|shutdown|debt ceiling|regulation|policy|rule|ruling|pass|vacancy|ban|tariff|ceiling))",
            ),
            allowed_categories=("legal_regulatory", "global_politics", "us_economic", "fed_monetary_policy"),
            exclude_patterns=(r"epstein|testify|contempt|last week tonight|john oliver",),
            min_contracts=15,
            target_contracts=16,
            max_per_template=1,
            max_per_community=4,
            **FAST_RISK_DEFAULTS,
        ),
        ThematicBasketSpec(
            domain="POLITICAL",
            basket_code="ADIT-P3",
            basket_name="EU Breakup Risk",
            basket_weight=0.3010,
            include_patterns=(
                ThemePattern(r"\beu\b|european union|eurozone", 1.0),
                ThemePattern(r"brexit|grexit|frexit|italexit|eu breakup", 1.2),
                ThemePattern(r"euro\b|schengen|ecb", 0.8),
            ),
            direct_patterns=(
                ThemePattern(r"withdraw|breakup|dissolve|member exit|debt crisis|trade fracture", 1.2),
            ),
            required_patterns=(
                r"(?=.*(?:\beu\b|european union|eurozone|ecb|schengen|euro))(?=.*(?:withdraw|breakup|dissolve|member|inflation|gdp|rate|trade|deficit|debt))",
            ),
            allowed_categories=("global_politics", "us_economic", "fed_monetary_policy", "russia_ukraine"),
            exclude_patterns=(r"trump and putin meet",),
            min_contracts=15,
            target_contracts=16,
            max_per_template=1,
            max_per_community=4,
            max_per_event_family=2,
            **FAST_RISK_DEFAULTS,
        ),
        ThematicBasketSpec(
            domain="POLITICAL",
            basket_code="ADIT-P2",
            basket_name="EM Governance",
            basket_weight=0.1511,
            include_patterns=(
                ThemePattern(r"emerging market", 1.0),
                ThemePattern(
                    r"\bargentina\b|\bbrazil\b|\bindia\b|\bindonesia\b|\bmexico\b|\bnigeria\b|\bturkey\b|\bsouth africa\b|\bvietnam\b|\bpakistan\b|\bphilippines\b|\bmalaysia\b|\bcolombia\b",
                    1.0,
                ),
                ThemePattern(r"governance|policy|regulator|central bank", 0.6),
            ),
            direct_patterns=(
                ThemePattern(r"government|cabinet|policy|regulator|central bank|sovereign|prime minister|parliament", 1.1),
                ThemePattern(r"rate shock|default|out|removed|resign", 0.9),
            ),
            required_patterns=(
                r"(?=.*(?:argentina|brazil|india|indonesia|mexico|nigeria|turkey|south africa|vietnam|pakistan|philippines|malaysia|colombia|chile|peru|thailand))(?=.*(?:central bank|policy|regulator|governance|prime minister|parliament|government|cabinet|rate|finance minister|sovereign))",
            ),
            allowed_categories=("global_politics", "us_economic", "fed_monetary_policy", "legal_regulatory"),
            exclude_patterns=(r"swiss|ecb|federal reserve|\bfed\b|eurozone|european union",),
            min_contracts=15,
            target_contracts=18,
            max_per_template=2,
            max_per_community=4,
            max_per_event_family=3,
            **FAST_RISK_DEFAULTS,
        ),
    ]
    excluded_codes = {"ADIT-AI1", "ADIT-SP1"}
    return [spec for spec in specs if spec.basket_code not in excluded_codes]


def _build_stochastic_gap_overlay(g: pd.DataFrame, basket_code: str) -> list[float | None]:
    if g.empty:
        return []
    nav = pd.to_numeric(g.get("tradable_nav", g.get("basket_level")), errors="coerce")
    cash = pd.to_numeric(g.get("cash_weight"), errors="coerce").fillna(0.0)
    coverage = pd.to_numeric(g.get("price_coverage_share"), errors="coerce").fillna(1.0)
    gap_mask = nav.notna() & (cash >= 0.999999) & (coverage <= 0.0)
    overlay: list[float | None] = [None] * len(g)
    log_nav = np.log(nav.where(nav > 0))
    log_returns = log_nav.diff().replace([np.inf, -np.inf], np.nan)

    i = 0
    n = len(g)
    while i < n:
        if not bool(gap_mask.iloc[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(gap_mask.iloc[j + 1]):
            j += 1
        left = i - 1
        right = j + 1
        if left >= 0 and right < n and pd.notna(nav.iloc[left]) and pd.notna(nav.iloc[right]):
            start = max(float(nav.iloc[left]), 1e-6)
            end = max(float(nav.iloc[right]), 1e-6)
            span = j - i + 1
            left_slice = log_returns.iloc[max(1, left - 30): left + 1]
            right_slice = log_returns.iloc[right: min(n, right + 30)]
            local_sigma = float(pd.concat([left_slice, right_slice]).dropna().std(ddof=0))
            if not math.isfinite(local_sigma) or local_sigma < 0.005:
                local_sigma = 0.03
            seed_src = f"{basket_code}|{g.iloc[i]['rebalance_date']}|{g.iloc[j]['rebalance_date']}"
            seed = int.from_bytes(hashlib.sha256(seed_src.encode("utf-8")).digest()[:8], "big") % (2**32 - 1)
            rng = np.random.default_rng(seed)
            shocks = rng.normal(0.0, local_sigma * 0.25, span)
            brownian = np.cumsum(shocks)
            bridge_noise = brownian - (np.arange(1, span + 1) / (span + 1)) * brownian[-1]
            t = np.arange(1, span + 1) / (span + 1)
            path = ((1.0 - t) * math.log(start)) + (t * math.log(end)) + bridge_noise
            vals = np.exp(path)
            for offset, value in enumerate(vals):
                overlay[i + offset] = float(value)
        i = j + 1
    return overlay


def load_market_universe(
    processed_dir: str | Path,
    *,
    require_temporal_history: bool = True,
    min_temporal_coverage: float = 0.98,
) -> pd.DataFrame:
    processed = Path(processed_dir)
    mapping_path = processed / "ticker_mapping.parquet"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing mapping file: {mapping_path}")

    df = pd.read_parquet(mapping_path).copy()
    for col in ["market_id", "ticker_id", "ticker_name", "title", "end_date_parsed"]:
        if col not in df.columns:
            raise ValueError(f"{mapping_path} missing required column: {col}")

    df["market_id"] = df["market_id"].astype(str)
    df["ticker_id"] = df["ticker_id"].astype(str)
    df["ticker_name"] = df["ticker_name"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["title_l"] = df["title"].str.lower()
    df["end_date"] = pd.to_datetime(df["end_date_parsed"], errors="coerce")
    df = df.dropna(subset=["end_date"]).copy()
    df["platform"] = np.where(df["market_id"].str.startswith("poly_"), "polymarket", "kalshi")
    df["listed_at"] = pd.NaT
    df["inactive_at"] = pd.NaT
    df["temporal_source"] = "none"
    if "event_slug" not in df.columns:
        df["event_slug"] = ""
    df["event_slug"] = df["event_slug"].fillna("").astype(str)

    # Optional live-ish price enrichment when normalized markets are present.
    markets_path = processed / "markets.parquet"
    if markets_path.exists():
        mk = pd.read_parquet(markets_path).copy()
        id_col = None
        for c in ("market_id", "contract_id", "id"):
            if c in mk.columns:
                id_col = c
                break
        px_col = None
        for c in ("current_price", "price", "last_price", "yes_price_dollars"):
            if c in mk.columns:
                px_col = c
                break
        if id_col is not None and px_col is not None:
            pmap = (
                mk[[id_col, px_col]]
                .dropna(subset=[id_col])
                .drop_duplicates(id_col, keep="last")
                .set_index(id_col)[px_col]
            )
            pmap.index = pmap.index.astype(str)
            df["current_price"] = pd.to_numeric(df["market_id"].map(pmap), errors="coerce")

        if id_col is not None:
            mk_id = mk[id_col].astype(str)
            mk_meta = pd.DataFrame(
                {
                    "market_id": mk_id,
                    "listed_at_markets": pd.to_datetime(
                        _first_existing_series(
                            mk,
                            ["created_at", "createdAt", "created_date", "start_date", "active_start", "startDate", "start_time"],
                        ),
                        errors="coerce",
                        utc=True,
                    ),
                    "inactive_at_markets": pd.to_datetime(
                        _first_existing_series(
                            mk,
                            ["resolution_date", "resolved_at", "resolvedAt", "active_end", "close_time", "closedTime"],
                        ),
                        errors="coerce",
                        utc=True,
                    ),
                }
            )
            mk_meta = mk_meta.sort_values(["market_id", "listed_at_markets"]).drop_duplicates("market_id", keep="first")
            df = df.merge(mk_meta, on="market_id", how="left")
            df["listed_at"] = _rowwise_min_datetime(df["listed_at"], df["listed_at_markets"])
            df["inactive_at"] = _rowwise_max_datetime(df["inactive_at"], df["inactive_at_markets"])
            mask_mk = pd.to_datetime(df["listed_at_markets"], errors="coerce", utc=True).notna()
            df.loc[mask_mk & (df["temporal_source"] == "none"), "temporal_source"] = "markets_metadata"
            df = df.drop(columns=["listed_at_markets", "inactive_at_markets"], errors="ignore")
    if "current_price" not in df.columns:
        df["current_price"] = np.nan

    # Optional full-history temporal registry (resumable backfill output).
    history_path = processed / "polymarket_market_history.parquet"
    history_loaded = False
    if history_path.exists():
        hist = pd.read_parquet(history_path).copy()
        if "market_id" in hist.columns:
            hist["market_id"] = hist["market_id"].astype(str)
            hist["listed_at_history"] = pd.to_datetime(
                _first_existing_series(hist, ["created_at", "first_seen_at"]),
                errors="coerce",
                utc=True,
            )
            hist["inactive_at_history"] = pd.to_datetime(
                _first_existing_series(hist, ["resolution_time", "closed_time"]),
                errors="coerce",
                utc=True,
            )
            keep_cols = ["market_id", "listed_at_history", "inactive_at_history"]
            df = df.merge(hist[keep_cols].drop_duplicates("market_id", keep="last"), on="market_id", how="left")
            df["listed_at"] = _rowwise_min_datetime(df["listed_at"], df["listed_at_history"])
            df["inactive_at"] = _rowwise_max_datetime(df["inactive_at"], df["inactive_at_history"])
            mask_hist = pd.to_datetime(df["listed_at_history"], errors="coerce", utc=True).notna()
            df.loc[mask_hist, "temporal_source"] = "polymarket_history"
            df = df.drop(columns=["listed_at_history", "inactive_at_history"], errors="ignore")
            history_loaded = True

    # Optional first print from prices (useful when metadata has sparse listing dates).
    prices_path = processed / "prices.parquet"
    if prices_path.exists():
        try:
            px = pd.read_parquet(prices_path, columns=["market_id", "date", "close_price"])
            if not px.empty and "market_id" in px.columns and "date" in px.columns:
                px["market_id"] = px["market_id"].astype(str)
                px = px.assign(date=pd.to_datetime(px["date"], errors="coerce", utc=True))
                first_px = (
                    px
                    .dropna(subset=["date"])
                    .groupby("market_id")["date"]
                    .min()
                    .rename("listed_at_price")
                )
                df["listed_at_price"] = pd.to_datetime(df["market_id"].map(first_px), errors="coerce", utc=True)
                df["listed_at"] = _rowwise_min_datetime(df["listed_at"], df["listed_at_price"])
                mask_px = pd.to_datetime(df["listed_at_price"], errors="coerce", utc=True).notna()
                df.loc[mask_px & (df["temporal_source"] == "none"), "temporal_source"] = "price_first_print"
                df = df.drop(columns=["listed_at_price"], errors="ignore")

                if "close_price" in px.columns:
                    px["close_price"] = pd.to_numeric(px["close_price"], errors="coerce")
                    latest_px = (
                        px.dropna(subset=["date", "close_price"])
                        .sort_values(["market_id", "date"])
                        .groupby("market_id")["close_price"]
                        .last()
                    )
                    missing_current = pd.to_numeric(df["current_price"], errors="coerce").isna()
                    df.loc[missing_current, "current_price"] = pd.to_numeric(
                        df.loc[missing_current, "market_id"].map(latest_px),
                        errors="coerce",
                    )
        except Exception:
            # Best-effort enrichment only.
            pass

    # Optional category enrichment from exact-title legacy cache.
    llm_cat_path = processed / "llm_market_categories.json"
    df["llm_category"] = "unknown"
    df["llm_secondary_category"] = ""
    df["classification_source"] = "unknown"
    df["classification_confidence"] = 0.0
    if llm_cat_path.exists():
        with open(llm_cat_path, "r", encoding="utf-8") as f:
            llm_cats = json.load(f)
        title_map = {str(k): str(v) for k, v in (llm_cats or {}).items()}
        title_l_map = {str(k).lower(): str(v) for k, v in (llm_cats or {}).items()}
        cached_category = (
            df["title"].map(title_map)
            .fillna(df["title_l"].map(title_l_map))
            .fillna("unknown")
        )
        cached_mask = cached_category.astype(str).str.lower().ne("unknown")
        df.loc[cached_mask, "llm_category"] = cached_category.loc[cached_mask].map(_normalize_builder_category)
        df.loc[cached_mask, "classification_source"] = "llm_title_cache"
        df.loc[cached_mask, "classification_confidence"] = 0.55

    # Optional row-level LLM classifications with market_id joins.
    classifications_path = processed / "market_classifications.parquet"
    if classifications_path.exists():
        cls = pd.read_parquet(classifications_path).copy()
        cls_id_col = next((c for c in ("market_id", "contract_id", "id") if c in cls.columns), None)
        if cls_id_col is not None and "primary_theme" in cls.columns:
            cls["market_id"] = cls[cls_id_col].astype(str)
            cls["primary_theme"] = cls["primary_theme"].map(_normalize_builder_category)
            if "secondary_theme" in cls.columns:
                cls["secondary_theme"] = cls["secondary_theme"].map(_normalize_builder_category)
            else:
                cls["secondary_theme"] = ""
            cls["confidence"] = pd.to_numeric(cls.get("confidence", 0.75), errors="coerce").fillna(0.75)
            cls = (
                cls.sort_values(["market_id", "confidence"], ascending=[True, False])
                .drop_duplicates("market_id", keep="first")
                [["market_id", "primary_theme", "secondary_theme", "confidence"]]
            )
            df = df.merge(cls, on="market_id", how="left")
            primary_mask = df["primary_theme"].fillna("").astype(str).str.len() > 0
            secondary_mask = df["secondary_theme"].fillna("").astype(str).str.len() > 0
            df.loc[primary_mask, "llm_category"] = df.loc[primary_mask, "primary_theme"].astype(str)
            df.loc[secondary_mask, "llm_secondary_category"] = df.loc[secondary_mask, "secondary_theme"].astype(str)
            df.loc[primary_mask, "classification_source"] = "market_classifications"
            df.loc[primary_mask, "classification_confidence"] = pd.to_numeric(df.loc[primary_mask, "confidence"], errors="coerce").fillna(0.75)
            df = df.drop(columns=["primary_theme", "secondary_theme", "confidence"], errors="ignore")

    # Optional LLM exposure / side cache keyed by market_id.
    exposure_path = processed / "exposure_classifications.json"
    df["exposure_direction"] = ""
    df["exposure_description"] = ""
    df["exposure_confidence"] = 0.0
    df["yes_outcome_polarity"] = ""
    df["yes_outcome_reason"] = ""
    df["direction_model"] = ""
    if exposure_path.exists():
        with open(exposure_path, "r", encoding="utf-8") as f:
            exposure_raw = json.load(f)
        if isinstance(exposure_raw, dict) and exposure_raw:
            exp_df = pd.DataFrame(
                [
                    {
                        "market_id": str(mid),
                        "exposure_direction": str((row or {}).get("exposure_direction", "")).lower(),
                        "exposure_description": str((row or {}).get("exposure_description", "")),
                        "exposure_confidence": float((row or {}).get("confidence", 0.0) or 0.0),
                        "yes_outcome_polarity": str((row or {}).get("yes_outcome_polarity", "")).lower(),
                        "yes_outcome_reason": str((row or {}).get("yes_outcome_reason", "")),
                        "direction_model": str((row or {}).get("model", "")),
                    }
                    for mid, row in exposure_raw.items()
                    if isinstance(row, dict)
                ]
            )
            if not exp_df.empty:
                exp_df = exp_df.sort_values(["market_id", "exposure_confidence"], ascending=[True, False]).drop_duplicates("market_id", keep="first")
                df = df.drop(
                    columns=[
                        "exposure_direction",
                        "exposure_description",
                        "exposure_confidence",
                        "yes_outcome_polarity",
                        "yes_outcome_reason",
                        "direction_model",
                    ],
                    errors="ignore",
                ).merge(exp_df, on="market_id", how="left")
                df["exposure_direction"] = df["exposure_direction"].fillna("").astype(str).str.lower()
                df["exposure_description"] = df["exposure_description"].fillna("").astype(str)
                df["exposure_confidence"] = pd.to_numeric(df["exposure_confidence"], errors="coerce").fillna(0.0)
                df["yes_outcome_polarity"] = df["yes_outcome_polarity"].fillna("").astype(str).str.lower()
                df["yes_outcome_reason"] = df["yes_outcome_reason"].fillna("").astype(str)
                df["direction_model"] = df["direction_model"].fillna("").astype(str)

    # Optional quality enrichment from strict clustering volume.
    strict_path = processed / "strict_clustering_results.json"
    volume_map: dict[str, float] = {}
    if strict_path.exists():
        with open(strict_path, "r", encoding="utf-8") as f:
            strict = json.load(f)
        for _, comm in (strict.get("communities") or {}).items():
            for m in comm.get("markets", []):
                mid = str(m.get("market_id", ""))
                if mid:
                    volume_map[mid] = float(m.get("volume", 0.0) or 0.0)
    df["volume"] = df["market_id"].map(volume_map).fillna(0.0)

    # Optional quality enrichment from correlation diagnostics.
    corr_path = processed / "correlation_clustering_results.json"
    timeseries_stats_path = processed / "ticker_timeseries_stats.json"
    obs_map: dict[str, float] = {}
    corr_count_map: dict[str, float] = {}
    std_change_map: dict[str, float] = {}
    mean_change_map: dict[str, float] = {}
    community_map: dict[str, int] = {}
    community_factor_map: dict[int, dict[str, float]] = {}
    if corr_path.exists():
        with open(corr_path, "r", encoding="utf-8") as f:
            corr_data = json.load(f)
        stats = corr_data.get("market_stats") or {}
        for mid, row in stats.items():
            obs_map[str(mid)] = float(row.get("n_observations", 0.0) or 0.0)
            corr_count_map[str(mid)] = float(row.get("valid_correlations", 0.0) or 0.0)
            std_change_map[str(mid)] = float(row.get("std_change", 0.0) or 0.0)
            mean_change_map[str(mid)] = float(row.get("mean_change", 0.0) or 0.0)

        assignments = corr_data.get("community_assignments") or {}
        for mid, cid in assignments.items():
            try:
                community_map[str(mid)] = int(cid)
            except Exception:
                continue

        cfac = corr_data.get("community_factors") or {}
        for cid, row in cfac.items():
            try:
                cid_int = int(cid)
            except Exception:
                continue
            mb = (row or {}).get("mean_betas") or {}
            community_factor_map[cid_int] = {str(k): float(v) for k, v in mb.items() if v is not None}

    ts_end = pd.NaT
    if timeseries_stats_path.exists():
        try:
            with open(timeseries_stats_path, "r", encoding="utf-8") as f:
                ts_stats = json.load(f)
            ts_end = pd.to_datetime((ts_stats or {}).get("date_range_end"), errors="coerce")
        except Exception:
            ts_end = pd.NaT

    df["n_observations"] = df["market_id"].map(obs_map).fillna(0.0)
    df["valid_correlations"] = df["market_id"].map(corr_count_map).fillna(0.0)
    df["std_change"] = df["market_id"].map(std_change_map).fillna(0.0)
    df["mean_change"] = df["market_id"].map(mean_change_map).fillna(0.0)
    df["community_id"] = df["market_id"].map(community_map).fillna(-1).astype(int)
    if pd.notna(ts_end):
        nobs = pd.to_numeric(df["n_observations"], errors="coerce").fillna(0.0)
        df["first_seen_proxy"] = pd.NaT
        mask_obs = nobs > 0
        if mask_obs.any():
            df.loc[mask_obs, "first_seen_proxy"] = pd.to_datetime(ts_end) - pd.to_timedelta(nobs[mask_obs] - 1, unit="D")
    else:
        df["first_seen_proxy"] = pd.NaT

    # Lowest-confidence temporal fallback for rows still missing listed_at.
    fs_proxy = pd.to_datetime(df["first_seen_proxy"], errors="coerce", utc=True)
    df["listed_at"] = _rowwise_min_datetime(df["listed_at"], fs_proxy)
    mask_proxy = fs_proxy.notna() & (df["temporal_source"] == "none")
    df.loc[mask_proxy, "temporal_source"] = "nobs_proxy"

    factor_keys = ["SPY", "QQQ", "GLD", "TLT", "TNX", "IRX", "VIX", "USO", "BTC_USD", "SHY", "TLH", "TYX", "FVX"]
    for fk in factor_keys:
        df[f"beta_{fk}"] = 0.0
    if community_factor_map:
        for cid, betas in community_factor_map.items():
            mask = df["community_id"] == cid
            if not mask.any():
                continue
            for fk in factor_keys:
                df.loc[mask, f"beta_{fk}"] = float(betas.get(fk, 0.0))

    # Composite quality proxy in [0, 1] via rank scaling.
    q1 = _rank_pct(np.log1p(df["volume"]))
    q2 = _rank_pct(df["n_observations"])
    q3 = _rank_pct(df["valid_correlations"])
    df["quality_score"] = 0.45 * q1 + 0.30 * q2 + 0.25 * q3
    # Risk/volatility proxy from observed daily change variability.
    vol = pd.to_numeric(df["std_change"], errors="coerce").fillna(0.0).replace(0, np.nan)
    fallback_vol = float(vol.dropna().median()) if vol.notna().any() else 0.02
    df["risk_vol_proxy"] = vol.fillna(fallback_vol).clip(lower=1e-4)

    # Temporal quality guardrail: without reliable listed_at, historical rebalances
    # can leak future contracts. Enforce coverage for Polymarket rows.
    poly_mask = df["platform"] == "polymarket"
    poly_count = int(poly_mask.sum())
    poly_temporal_coverage = float(pd.to_datetime(df.loc[poly_mask, "listed_at"], errors="coerce", utc=True).notna().mean()) if poly_count else 1.0
    if require_temporal_history and poly_count and (poly_temporal_coverage < float(min_temporal_coverage)):
        history_msg = (
            f"Found {history_path}" if history_loaded else f"Missing or incomplete {history_path}"
        )
        raise RuntimeError(
            "Insufficient Polymarket listing-date coverage for strict historical rebalance filtering "
            f"({poly_temporal_coverage:.1%} < required {float(min_temporal_coverage):.1%}). {history_msg}. "
            "Refresh data/processed/polymarket_market_history.parquet with fuller listing-date coverage, "
            "then regenerate baskets."
        )

    df["listed_at"] = pd.to_datetime(df["listed_at"], errors="coerce", utc=True).dt.tz_convert(None)
    df["inactive_at"] = pd.to_datetime(df["inactive_at"], errors="coerce", utc=True).dt.tz_convert(None)
    df["first_seen_proxy"] = pd.to_datetime(df["first_seen_proxy"], errors="coerce", utc=True).dt.tz_convert(None)
    df["temporal_coverage"] = poly_temporal_coverage

    # Keep a single row per market id.
    df = df.sort_values(["market_id", "end_date"]).drop_duplicates("market_id", keep="last")
    return _apply_semantic_heuristics(df.reset_index(drop=True))


def load_ticker_chains(processed_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = Path(processed_dir)
    mapping_path = processed / "ticker_mapping.parquet"
    if mapping_path.exists():
        cols = [
            "ticker_id",
            "ticker_name",
            "market_id",
            "title",
            "event_slug",
            "end_date",
        ]
        mapping = pd.read_parquet(mapping_path, columns=cols).copy()
        if not mapping.empty and "ticker_id" in mapping.columns:
            mapping["ticker_id"] = mapping["ticker_id"].astype(str)
            mapping["ticker_name"] = mapping.get("ticker_name", "").fillna("").astype(str)
            mapping["market_id"] = mapping.get("market_id", "").fillna("").astype(str)
            mapping["title"] = mapping.get("title", "").fillna("").astype(str)
            mapping["event_slug"] = mapping.get("event_slug", "").fillna("").astype(str)
            mapping["end_date"] = pd.to_datetime(mapping.get("end_date"), errors="coerce")
            mapping = mapping[(mapping["ticker_id"] != "") & (mapping["market_id"] != "")].copy()
            if not mapping.empty:
                markets_df = (
                    mapping.sort_values(["ticker_id", "end_date", "market_id"])
                    .drop_duplicates(["ticker_id", "market_id"], keep="last")
                    .reset_index(drop=True)
                )
                chain_df = (
                    markets_df.groupby(["ticker_id", "ticker_name"], dropna=False)
                    .agg(
                        chain_market_count=("market_id", "nunique"),
                        chain_first_end_date=("end_date", "min"),
                        chain_last_end_date=("end_date", "max"),
                        chain_event_slugs=("event_slug", lambda s: ", ".join(sorted({str(x).strip() for x in s if str(x).strip()}))),
                    )
                    .reset_index()
                )
                for col in ["chain_first_end_date", "chain_last_end_date"]:
                    chain_df[col] = pd.to_datetime(chain_df[col], errors="coerce").dt.date.astype(str).replace("NaT", "")
                return chain_df, markets_df

    chains_path = processed / "ticker_chains.json"
    if not chains_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    with open(chains_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    chain_rows: list[dict] = []
    market_rows: list[dict] = []
    for _, row in (raw or {}).items():
        if not isinstance(row, dict):
            continue
        ticker_id = str(row.get("ticker_id", "")).strip()
        if not ticker_id:
            continue
        ticker_name = str(row.get("ticker_name", ""))
        markets = row.get("markets") or []
        ends: list[pd.Timestamp] = []
        slugs: set[str] = set()
        for m in markets:
            if not isinstance(m, dict):
                continue
            mid = str(m.get("market_id", "")).strip()
            if not mid:
                continue
            end_raw = m.get("end_date")
            end_ts = pd.to_datetime(end_raw, errors="coerce")
            if pd.notna(end_ts):
                ends.append(pd.Timestamp(end_ts))
            es = str(m.get("event_slug", "")).strip()
            if es:
                slugs.add(es)
            market_rows.append(
                {
                    "ticker_id": ticker_id,
                    "ticker_name": ticker_name,
                    "chain_market_count": int(row.get("market_count", 0) or 0),
                    "market_id": mid,
                    "title": str(m.get("title", "")),
                    "end_date": end_ts,
                    "event_slug": es,
                }
            )
        chain_rows.append(
            {
                "ticker_id": ticker_id,
                "ticker_name": ticker_name,
                "chain_market_count": int(row.get("market_count", 0) or 0),
                "chain_first_end_date": min(ends).date().isoformat() if ends else "",
                "chain_last_end_date": max(ends).date().isoformat() if ends else "",
                "chain_event_slugs": ", ".join(sorted(slugs)),
            }
        )

    chain_df = pd.DataFrame(chain_rows).drop_duplicates("ticker_id", keep="first")
    markets_df = pd.DataFrame(market_rows)
    if not markets_df.empty:
        markets_df["end_date"] = pd.to_datetime(markets_df["end_date"], errors="coerce")
        markets_df = markets_df.sort_values(["ticker_id", "end_date", "market_id"]).reset_index(drop=True)
    return chain_df, markets_df


class ThematicBasketBuilder:
    def __init__(
        self,
        universe: pd.DataFrame,
        specs: list[ThematicBasketSpec] | None = None,
        min_days_to_expiry: int = 21,
        certainty_low: float = 0.05,
        certainty_high: float = 0.95,
        strict_temporal: bool = True,
    ):
        if universe.empty:
            raise ValueError("universe is empty")
        self.universe = universe.copy()
        # Backward-compatible defaults for synthetic tests / reduced inputs.
        if "risk_vol_proxy" not in self.universe.columns:
            self.universe["risk_vol_proxy"] = 0.02
        if "mean_change" not in self.universe.columns:
            self.universe["mean_change"] = 0.0
        if "community_id" not in self.universe.columns:
            self.universe["community_id"] = -1
        if "quality_score" not in self.universe.columns:
            self.universe["quality_score"] = 0.5
        if "current_price" not in self.universe.columns:
            self.universe["current_price"] = np.nan
        if "first_seen_proxy" not in self.universe.columns:
            self.universe["first_seen_proxy"] = pd.NaT
        if "listed_at" not in self.universe.columns:
            self.universe["listed_at"] = pd.NaT
        if "inactive_at" not in self.universe.columns:
            self.universe["inactive_at"] = pd.NaT
        if "temporal_source" not in self.universe.columns:
            self.universe["temporal_source"] = "none"
        if "llm_category" not in self.universe.columns:
            self.universe["llm_category"] = "unknown"
        if "llm_secondary_category" not in self.universe.columns:
            self.universe["llm_secondary_category"] = ""
        if "classification_source" not in self.universe.columns:
            self.universe["classification_source"] = "unknown"
        if "classification_confidence" not in self.universe.columns:
            self.universe["classification_confidence"] = 0.0
        if "exposure_direction" not in self.universe.columns:
            self.universe["exposure_direction"] = ""
        if "exposure_description" not in self.universe.columns:
            self.universe["exposure_description"] = ""
        if "exposure_confidence" not in self.universe.columns:
            self.universe["exposure_confidence"] = 0.0
        if "yes_outcome_polarity" not in self.universe.columns:
            self.universe["yes_outcome_polarity"] = ""
        if "yes_outcome_reason" not in self.universe.columns:
            self.universe["yes_outcome_reason"] = ""
        if "direction_model" not in self.universe.columns:
            self.universe["direction_model"] = ""
        if "event_slug" not in self.universe.columns:
            self.universe["event_slug"] = ""
        needs_semantic_pass = (
            ("search_text" not in self.universe.columns)
            or self.universe["llm_category"].isna().any()
            or self.universe["classification_source"].isna().any()
        )
        if needs_semantic_pass:
            self.universe = _apply_semantic_heuristics(self.universe)
        else:
            self.universe["llm_category"] = self.universe["llm_category"].map(_normalize_builder_category).fillna("unknown")
            self.universe["llm_secondary_category"] = self.universe["llm_secondary_category"].map(_normalize_builder_category).fillna("")
            self.universe["classification_source"] = self.universe["classification_source"].fillna("unknown").astype(str)
            self.universe["classification_confidence"] = pd.to_numeric(self.universe["classification_confidence"], errors="coerce").fillna(0.0)
            self.universe["exposure_direction"] = self.universe["exposure_direction"].fillna("").astype(str).str.lower()
            self.universe["exposure_description"] = self.universe["exposure_description"].fillna("").astype(str)
            self.universe["exposure_confidence"] = pd.to_numeric(self.universe["exposure_confidence"], errors="coerce").fillna(0.0)
            self.universe["yes_outcome_polarity"] = self.universe["yes_outcome_polarity"].fillna("").astype(str).str.lower()
            self.universe["yes_outcome_reason"] = self.universe["yes_outcome_reason"].fillna("").astype(str)
            self.universe["direction_model"] = self.universe["direction_model"].fillna("").astype(str)
        self.specs = specs or default_specs()
        self.min_days_to_expiry = int(min_days_to_expiry)
        self.certainty_low = float(certainty_low)
        self.certainty_high = float(certainty_high)
        self.strict_temporal = bool(strict_temporal)

        self._global_exclude_re = _compile_union(GLOBAL_EXCLUDE_PATTERNS)
        self.last_slot_coverage_diagnostics = pd.DataFrame()

    def _theme_keyword_score(self, titles: pd.Series, spec: ThematicBasketSpec) -> pd.Series:
        return _score_text_patterns(titles, spec.include_patterns)

    def _filter_theme_candidates(self, as_of: pd.Timestamp, spec: ThematicBasketSpec) -> pd.DataFrame:
        df = self.universe.copy()
        df["days_to_expiry"] = (df["end_date"] - as_of).dt.days
        df = df[df["days_to_expiry"] >= self.min_days_to_expiry].copy()
        # Temporal availability control: contract must be listed by as_of.
        listed = pd.to_datetime(
            (df["listed_at"] if "listed_at" in df.columns else pd.Series(pd.NaT, index=df.index)),
            errors="coerce",
        )
        if listed.notna().any():
            if self.strict_temporal:
                # Strict mode disallows unknown listing timestamps.
                df = df[listed.notna()].copy()
                listed = pd.to_datetime(
                    (df["listed_at"] if "listed_at" in df.columns else pd.Series(pd.NaT, index=df.index)),
                    errors="coerce",
                )
            df = df[listed <= as_of].copy()
        elif self.strict_temporal:
            # If no explicit listed_at, fall back to first_seen proxy only in strict mode.
            fs = pd.to_datetime(
                (df["first_seen_proxy"] if "first_seen_proxy" in df.columns else pd.Series(pd.NaT, index=df.index)),
                errors="coerce",
            )
            df = df[fs.notna() & (fs <= as_of)].copy()

        inactive = pd.to_datetime(
            (df["inactive_at"] if "inactive_at" in df.columns else pd.Series(pd.NaT, index=df.index)),
            errors="coerce",
        )
        if inactive.notna().any():
            # If a contract was already resolved/closed before as_of, exclude it.
            df = df[inactive.isna() | (inactive >= as_of)].copy()

        if self._global_exclude_re is not None:
            df = df[~df["search_text"].str.contains(self._global_exclude_re, na=False)].copy()

        if spec.exclude_patterns:
            excl = _compile_union(spec.exclude_patterns)
            if excl is not None:
                df = df[~df["search_text"].str.contains(excl, na=False)].copy()

        df_base = df.copy()

        if spec.required_patterns:
            req = _compile_union(spec.required_patterns)
            if req is not None:
                df = df[df["search_text"].str.contains(req, na=False)].copy()

        df["keyword_score"] = self._theme_keyword_score(df["search_text"], spec)
        df = df[df["keyword_score"] > 0].copy()
        if df.empty:
            df = df_base.copy()
            df["keyword_score"] = self._theme_keyword_score(df["search_text"], spec)
            df = df[df["keyword_score"] > 0].copy()
            if df.empty:
                return df

        # Taxonomy gate: keep category-aligned contracts and only allow uncertain "other/unknown"
        # rows if thematic keyword score is high enough.
        def _apply_category_gate(frame: pd.DataFrame, relax: bool = False) -> pd.DataFrame:
            x = frame.copy()
            x["llm_category"] = x["llm_category"].fillna("unknown").astype(str)
            secondary = x["llm_secondary_category"].fillna("").astype(str) if "llm_secondary_category" in x.columns else pd.Series("", index=x.index)
            source = x["classification_source"].fillna("unknown").astype(str) if "classification_source" in x.columns else pd.Series("unknown", index=x.index)
            if not spec.allowed_categories:
                x["category_bonus"] = 0.0
                return x

            allowed = {str(v) for v in spec.allowed_categories}
            cat_allowed_primary = x["llm_category"].isin(allowed)
            cat_allowed_secondary = secondary.isin(allowed)
            cat_allowed = cat_allowed_primary | cat_allowed_secondary
            fallback_unknown = x["llm_category"].isin(_CATEGORY_FALLBACK) & (x["keyword_score"] >= (1.0 if relax else 1.4))
            spillover = (~x["llm_category"].isin(allowed | _CATEGORY_FALLBACK)) & (x["keyword_score"] >= (1.8 if relax else 2.2))
            keep_mask = cat_allowed | fallback_unknown | spillover
            x = x[keep_mask].copy()
            if x.empty:
                return x
            primary_allowed_now = x["llm_category"].isin(allowed)
            secondary_allowed_now = secondary.reindex(x.index).isin(allowed)
            heuristic_allowed = source.reindex(x.index).str.startswith("heuristic")
            x["category_bonus"] = np.where(
                primary_allowed_now & heuristic_allowed,
                0.75,
                np.where(
                    primary_allowed_now,
                    1.0,
                    np.where(secondary_allowed_now, 0.65, (-0.10 if relax else -0.35)),
                ),
            )
            return x

        df = _apply_category_gate(df, relax=False)
        if df.empty or len(df) < spec.min_contracts:
            fallback = df_base.copy()
            fallback["keyword_score"] = self._theme_keyword_score(fallback["search_text"], spec)
            fallback = fallback[fallback["keyword_score"] >= 1.0].copy()
            fallback = _apply_category_gate(fallback, relax=True)
            if not fallback.empty:
                df = (
                    pd.concat([df, fallback], ignore_index=True)
                    .sort_values("keyword_score", ascending=False)
                    .drop_duplicates("market_id", keep="first")
                )
        if df.empty:
            return df

        direct_patterns = spec.direct_patterns or spec.include_patterns
        inverse_patterns = spec.inverse_patterns or ()
        df["direct_score"] = _score_text_patterns(df["search_text"], direct_patterns)
        df["inverse_score"] = _score_text_patterns(df["search_text"], inverse_patterns)
        rel = df["direct_score"] - df["inverse_score"]
        exposure_dir = df["exposure_direction"].fillna("").astype(str).str.lower()
        outcome_polarity = df["yes_outcome_polarity"].fillna("").astype(str).str.lower()
        outcome_confidence = pd.to_numeric(df.get("exposure_confidence", 0.0), errors="coerce").fillna(0.0)
        side = pd.Series("YES", index=df.index, dtype=object)
        side_source = pd.Series("default_yes", index=df.index, dtype=object)
        pattern_no = rel < -0.05
        pattern_yes = rel > 0.05
        side.loc[pattern_no] = "NO"
        side_source.loc[pattern_no] = "pattern_inverse"
        side.loc[pattern_yes] = "YES"
        side_source.loc[pattern_yes] = "pattern_direct"

        llm_threshold = 0.70
        llm_mask = outcome_confidence >= llm_threshold
        if spec.theme_polarity == "risk_up":
            llm_yes = llm_mask & outcome_polarity.eq("risk_up")
            llm_no = llm_mask & outcome_polarity.eq("risk_down")
        elif spec.theme_polarity == "growth_up":
            llm_yes = llm_mask & outcome_polarity.eq("growth_up")
            llm_no = llm_mask & outcome_polarity.eq("growth_down")
        else:
            llm_yes = pd.Series(False, index=df.index)
            llm_no = pd.Series(False, index=df.index)

        side.loc[llm_yes] = "YES"
        side_source.loc[llm_yes] = outcome_polarity.loc[llm_yes].map(lambda v: f"llm_theme_{v}")
        side.loc[llm_no] = "NO"
        side_source.loc[llm_no] = outcome_polarity.loc[llm_no].map(lambda v: f"llm_theme_{v}")

        if spec.theme_polarity in {"risk_up", "growth_up"}:
            exp_mask = exposure_dir.isin({"long", "short"}) & (rel.abs() <= 0.60) & ~(llm_yes | llm_no)
            if exp_mask.any():
                desired_side = np.where(
                    exposure_dir.eq("short") if spec.theme_polarity == "risk_up" else exposure_dir.eq("long"),
                    "YES",
                    "NO",
                )
                desired_side = pd.Series(desired_side, index=df.index)
                side.loc[exp_mask] = desired_side.loc[exp_mask]
                side_source.loc[exp_mask] = "exposure_cache"

        df["position_side"] = side.astype(str)
        df["side_source"] = side_source.astype(str)
        df["position_instruction"] = df["position_side"].map(_make_position_instruction)
        df["theme_side_score"] = np.maximum(df["direct_score"], df["inverse_score"]).astype(float)

        price_col = None
        for c in ("current_price", "price", "last_price"):
            if c in df.columns:
                price_col = c
                break
        if price_col is not None:
            raw_p = pd.to_numeric(df[price_col], errors="coerce")
            eff_p = raw_p.where(df["position_side"].ne("NO"), 1.0 - raw_p)
            df["market_yes_price"] = raw_p.clip(lower=0.0, upper=1.0)
            df["effective_price"] = eff_p.clip(lower=0.0, upper=1.0)
        else:
            df["market_yes_price"] = np.nan
            df["effective_price"] = np.nan
        df["effective_risk_price"] = df["effective_price"]
        df["preferred_probability_band"] = df["effective_risk_price"].map(
            lambda p: _classify_probability_band(
                p,
                certainty_low=self.certainty_low,
                certainty_high=self.certainty_high,
                preferred_min=spec.preferred_effective_price_min,
                preferred_max=spec.preferred_effective_price_max,
                soft_min=spec.soft_effective_price_min,
                soft_max=spec.soft_effective_price_max,
            )
        )
        df["probability_band_bonus"] = df["preferred_probability_band"].map(_probability_band_bonus).astype(float)
        df["direction_reason"] = [
            _direction_reason(st, ps, ss, ds, ins)
            for st, ps, ss, ds, ins in zip(
                df["search_text"],
                df["position_side"],
                df["side_source"],
                df["direct_score"],
                df["inverse_score"],
            )
        ]

        certainty_mask = (
            df["effective_risk_price"].notna()
            & (
                (df["effective_risk_price"] <= self.certainty_low)
                | (df["effective_risk_price"] >= self.certainty_high)
            )
        )
        df = df[~certainty_mask].copy()
        if df.empty:
            return df

        df["template_key"] = df["title_l"].map(_template_key_from_title)
        df["token_signature"] = df["title_l"].map(lambda t: " ".join(sorted(set(_tokenize_title(t)))))

        def _row_event_key(row: pd.Series) -> str:
            es = str(row.get("event_slug", "") or "").strip()
            if es and es.lower() != "nan":
                return _event_family_key(es, str(row.get("title_l", "")))
            return f"market-{row.get('market_id', '')}"

        df["event_family_key"] = df.apply(_row_event_key, axis=1)
        df["exclusive_group_key"] = df.apply(
            lambda row: _mutual_exclusion_key(str(row.get("event_slug", "")), str(row.get("title_l", ""))),
            axis=1,
        )
        df = _annotate_slot_fields(df, spec)
        df["tenor_target_days"] = int(spec.tenor_target_days)
        df["tenor_min_days"] = int(spec.tenor_min_days)
        df["tenor_max_days"] = int(spec.tenor_max_days)
        df["roll_floor_days"] = int(spec.roll_floor_days)
        df["spot_horizon_days"] = int(spec.spot_horizon_days)
        df["tenor_distance_days"] = (pd.to_numeric(df["days_to_expiry"], errors="coerce") - float(spec.tenor_target_days)).abs()
        df["tenor_band_status"] = df["days_to_expiry"].map(lambda v: _tenor_band_status(v, spec))
        df["tenor_band_bonus"] = df["tenor_band_status"].map(_tenor_band_bonus).astype(float)

        dte_target = max(int(spec.tenor_target_days or spec.target_dte_days), 1)
        dte_score = np.exp(-np.abs(df["days_to_expiry"] - dte_target) / dte_target)
        tenor_score = 1.0 - (pd.to_numeric(df["tenor_distance_days"], errors="coerce").fillna(dte_target) / float(dte_target)).clip(lower=0.0, upper=1.0)
        kw_z = _zscore(df["keyword_score"])
        q_z = _zscore(df["quality_score"])
        dte_z = _zscore(dte_score)
        tenor_z = _zscore(tenor_score)
        vol_z = _zscore(np.log1p(df["risk_vol_proxy"]))
        trend_z = _zscore(df["mean_change"])
        cat_z = _zscore(df["category_bonus"])
        stability = -vol_z
        side_z = _zscore(df["theme_side_score"])
        band_z = _zscore(df["probability_band_bonus"])
        tenor_band_z = _zscore(df["tenor_band_bonus"])
        slot_z = _zscore(df["slot_score"]) if spec.slot_schema else pd.Series(np.zeros(len(df)), index=df.index)

        df["selection_score"] = (
            0.28 * kw_z
            + 0.22 * q_z
            + float(spec.dte_score_weight) * dte_z
            + float(spec.tenor_score_weight) * tenor_z
            + float(spec.tenor_band_weight) * tenor_band_z
            + 0.10 * stability
            + 0.05 * trend_z
            + 0.05 * cat_z
            + 0.03 * side_z
            + 0.10 * band_z
            + 0.08 * slot_z
        )
        if spec.slot_schema:
            df.loc[df["slot_key"].eq("unassigned"), "selection_score"] = df.loc[df["slot_key"].eq("unassigned"), "selection_score"] - 0.85

        df = df.sort_values("selection_score", ascending=False)
        df = df.drop_duplicates("ticker_id", keep="first")
        return df

    def _select_contracts_generic(
        self,
        candidates: pd.DataFrame,
        spec: ThematicBasketSpec,
        *,
        target_override: int | None = None,
        min_override: int | None = None,
    ) -> pd.DataFrame:
        if candidates.empty:
            return candidates

        target = min(target_override if target_override is not None else spec.target_contracts, spec.max_contracts)
        n = min(len(candidates), target)
        min_contracts = spec.min_contracts if min_override is None else int(min_override)
        if n < min_contracts:
            n = min(len(candidates), min_contracts)

        pool = candidates.sort_values("selection_score", ascending=False).copy()
        if pool.empty:
            return pool

        factor_cols = [c for c in pool.columns if c.startswith("beta_")]
        selected_idx: list[int] = []
        template_counts: dict[str, int] = {}
        comm_counts: dict[int, int] = {}
        event_counts: dict[str, int] = {}
        exclusive_counts: dict[str, int] = {}
        selected_token_sets: list[set[str]] = []
        selected_factor_sum = np.zeros(len(factor_cols), dtype=float) if factor_cols else np.zeros(0, dtype=float)

        for relax_round in range(3):
            if len(selected_idx) >= n:
                break
            template_cap = spec.max_per_template + relax_round
            event_cap = spec.max_per_event_family + relax_round
            community_cap = spec.max_per_community + max(0, relax_round - 1)
            penalty_mult = 1.0 if relax_round == 0 else (0.82 if relax_round == 1 else 0.66)

            while len(selected_idx) < n:
                best_idx = None
                best_score = -1e18
                best_tokens: set[str] | None = None
                best_fvec: np.ndarray | None = None

                for idx, row in pool.iterrows():
                    if idx in selected_idx:
                        continue
                    tkey = str(row.get("template_key", "generic-template"))
                    if template_counts.get(tkey, 0) >= template_cap:
                        continue
                    ekey = str(row.get("event_family_key", "generic-event"))
                    if event_counts.get(ekey, 0) >= event_cap:
                        continue
                    gkey = str(row.get("exclusive_group_key", "")).strip()
                    if gkey and exclusive_counts.get(gkey, 0) >= spec.max_per_exclusive_group:
                        continue
                    cid = int(row.get("community_id", -1))
                    if cid >= 0 and comm_counts.get(cid, 0) >= community_cap:
                        continue

                    tok_set = set(str(row.get("token_signature", "")).split())
                    max_sim = max((_jaccard_similarity(tok_set, t) for t in selected_token_sets), default=0.0)
                    template_pen = template_counts.get(tkey, 0) * 0.22 * penalty_mult
                    event_pen = event_counts.get(ekey, 0) * 0.35 * penalty_mult
                    exclusive_pen = exclusive_counts.get(gkey, 0) * 0.55 if gkey else 0.0
                    comm_pen = comm_counts.get(cid, 0) * 0.12 * penalty_mult if cid >= 0 else 0.0

                    factor_pen = 0.0
                    fvec = None
                    if factor_cols:
                        fvec = row[factor_cols].to_numpy(dtype=float)
                        proj = (selected_factor_sum + fvec) / float(len(selected_idx) + 1)
                        treasury_idx = [i for i, c in enumerate(factor_cols) if c in {"beta_TNX", "beta_TLT", "beta_SHY", "beta_TLH", "beta_TYX", "beta_FVX", "beta_IRX"}]
                        broad_idx = [i for i, c in enumerate(factor_cols) if c in {"beta_SPY", "beta_QQQ", "beta_VIX", "beta_USO", "beta_GLD", "beta_BTC_USD"}]
                        tre = float(np.sum(np.abs(proj[treasury_idx]))) if treasury_idx else 0.0
                        broad = float(np.sum(np.abs(proj[broad_idx]))) if broad_idx else 0.0
                        factor_pen = (0.75 * tre + 0.45 * broad) * penalty_mult

                    marginal = float(row["selection_score"]) - 0.35 * max_sim - template_pen - event_pen - exclusive_pen - comm_pen - factor_pen
                    if marginal > best_score:
                        best_score = marginal
                        best_idx = idx
                        best_tokens = tok_set
                        best_fvec = fvec

                if best_idx is None:
                    break

                selected_idx.append(best_idx)
                sel_row = pool.loc[best_idx]
                tk = str(sel_row.get("template_key", "generic-template"))
                template_counts[tk] = template_counts.get(tk, 0) + 1
                ek = str(sel_row.get("event_family_key", "generic-event"))
                event_counts[ek] = event_counts.get(ek, 0) + 1
                gk = str(sel_row.get("exclusive_group_key", "")).strip()
                if gk:
                    exclusive_counts[gk] = exclusive_counts.get(gk, 0) + 1
                cid = int(sel_row.get("community_id", -1))
                if cid >= 0:
                    comm_counts[cid] = comm_counts.get(cid, 0) + 1
                selected_token_sets.append(best_tokens or set())
                if factor_cols and best_fvec is not None:
                    selected_factor_sum = selected_factor_sum + best_fvec

        selected = pool.loc[selected_idx].copy() if selected_idx else pd.DataFrame(columns=pool.columns)
        return selected.sort_values("selection_score", ascending=False)

    def _select_contracts(self, candidates: pd.DataFrame, spec: ThematicBasketSpec) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        if not spec.slot_schema:
            out = self._select_contracts_generic(candidates, spec)
            if not out.empty and "slot_selection_status" not in out.columns:
                out = out.copy()
                out["slot_selection_status"] = np.where(
                    out.get("preferred_probability_band", pd.Series("", index=out.index)).eq("tail"),
                    "tail_only",
                    "in_band",
                )
            return out

        pool = candidates.sort_values("selection_score", ascending=False).copy()
        selected_parts: list[pd.DataFrame] = []
        used_market_ids: set[str] = set()

        def _pick_tenor_pool(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
            if frame.empty or "tenor_band_status" not in frame.columns:
                return frame, "fallback"
            in_band = frame[frame["tenor_band_status"].astype(str) == "in_band"].copy()
            if not in_band.empty:
                return in_band, "in_band"
            non_in_band = frame[frame["tenor_band_status"].astype(str).isin(["above_band", "below_band"])].copy()
            if non_in_band.empty:
                return frame, "fallback"
            non_in_band["tenor_distance_days"] = pd.to_numeric(non_in_band["tenor_distance_days"], errors="coerce")
            band_choice = (
                non_in_band.groupby("tenor_band_status")["tenor_distance_days"]
                .median()
                .sort_values()
                .index.tolist()
            )
            chosen_band = str(band_choice[0]) if band_choice else "fallback"
            chosen = non_in_band[non_in_band["tenor_band_status"].astype(str) == chosen_band].copy()
            return (chosen if not chosen.empty else frame), chosen_band

        def _pick_probability_pool(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
            if frame.empty:
                return frame, "empty"
            preferred = frame[frame["preferred_probability_band"] == "preferred"].copy()
            soft = frame[frame["preferred_probability_band"] == "soft"].copy()
            tail = frame[frame["preferred_probability_band"] == "tail"].copy()
            if not preferred.empty:
                picked = pd.concat([preferred, soft], ignore_index=False).drop_duplicates("market_id", keep="first")
                return picked, "selected"
            if not soft.empty:
                return soft, "soft_only"
            return tail, "tail_only"

        for slot in spec.slot_schema:
            slot_pool = pool[pool["slot_key"].astype(str) == str(slot.slot_key)].copy()
            if slot_pool.empty:
                continue

            tenor_pool, tenor_pick = _pick_tenor_pool(slot_pool)
            candidate_pool, prob_pick = _pick_probability_pool(tenor_pool)
            slot_status = prob_pick if tenor_pick == "in_band" else f"{tenor_pick}_{prob_pick}"
            if candidate_pool.empty:
                continue

            slot_min = min(int(slot.min_names), len(candidate_pool))
            slot_target = min(int(slot.max_names), len(candidate_pool))
            slot_spec = replace(
                spec,
                min_contracts=max(0, slot_min),
                target_contracts=max(0, slot_target),
                max_contracts=max(0, slot_target),
            )
            slot_selected = self._select_contracts_generic(candidate_pool, slot_spec, target_override=slot_target, min_override=slot_min)
            if slot_status == "tail_only" and not slot_selected.empty:
                tail_cap = max(1, int(math.ceil(float(slot.max_names) * 0.5)))
                slot_selected = slot_selected.head(tail_cap).copy()
            if slot_selected.empty:
                continue
            slot_selected = slot_selected[~slot_selected["market_id"].astype(str).isin(used_market_ids)].copy()
            if slot_selected.empty:
                continue
            slot_selected["slot_selection_status"] = slot_status
            selected_parts.append(slot_selected)
            used_market_ids.update(slot_selected["market_id"].astype(str).tolist())

        selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=pool.columns)
        target = min(spec.target_contracts, spec.max_contracts)
        min_contracts = spec.min_contracts

        if len(selected) < target:
            remaining = pool[~pool["market_id"].astype(str).isin(used_market_ids)].copy()
            in_band_remaining = remaining[remaining["tenor_band_status"].astype(str) == "in_band"].copy() if "tenor_band_status" in remaining.columns else pd.DataFrame()
            fill_pref = in_band_remaining[in_band_remaining["preferred_probability_band"].isin(["preferred", "soft"])].copy() if not in_band_remaining.empty else pd.DataFrame()
            if not fill_pref.empty:
                fill_pool = fill_pref
            else:
                broader_pref = remaining[remaining["preferred_probability_band"].isin(["preferred", "soft"])].copy()
                fill_pool = broader_pref if not broader_pref.empty else (in_band_remaining if not in_band_remaining.empty else remaining)
            remaining_target = max(0, target - len(selected))
            remaining_min = max(0, min_contracts - len(selected))
            if remaining_target > 0 and not fill_pool.empty:
                fill_selected = self._select_contracts_generic(
                    fill_pool,
                    spec,
                    target_override=remaining_target,
                    min_override=remaining_min,
                )
                if not fill_selected.empty:
                    fill_selected = fill_selected.copy()
                    fill_selected["slot_selection_status"] = np.where(
                        fill_selected["preferred_probability_band"].eq("tail"),
                        "fallback_tail",
                        "fallback",
                    )
                    if selected.empty:
                        selected = fill_selected.reset_index(drop=True)
                    else:
                        selected = pd.concat([selected, fill_selected], ignore_index=True)

        if selected.empty:
            return selected
        selected = selected.drop_duplicates("market_id", keep="first").sort_values("selection_score", ascending=False)
        return selected.head(spec.max_contracts).reset_index(drop=True)

    def _weight_selected(self, selected: pd.DataFrame, spec: ThematicBasketSpec) -> pd.DataFrame:
        if selected.empty:
            return selected

        sel = selected.set_index("market_id")
        score_rank = _rank_pct(sel["selection_score"])
        quality_rank = _rank_pct(sel["quality_score"]) if "quality_score" in sel.columns else pd.Series(0.5, index=sel.index)
        vol_s = pd.to_numeric(sel["volume"], errors="coerce").fillna(0.0) if "volume" in sel.columns else pd.Series(0.0, index=sel.index)
        risk_s = pd.to_numeric(sel["risk_vol_proxy"], errors="coerce").fillna(0.02) if "risk_vol_proxy" in sel.columns else pd.Series(0.02, index=sel.index)
        liq_rank = _rank_pct(np.log1p(vol_s))
        stability_rank = _rank_pct(-np.log1p(risk_s))
        raw = 0.25 + 0.45 * score_rank + 0.22 * quality_rank + 0.18 * liq_rank + 0.15 * stability_rank
        w = _clip_and_normalize(raw, max_weight=spec.max_single_weight)

        if "slot_key" in sel.columns and spec.slot_schema:
            slot_caps = {slot.slot_key: float(slot.max_slot_weight) for slot in spec.slot_schema if float(slot.max_slot_weight or 0) > 0}
            w = _apply_slot_caps(w, sel["slot_key"], slot_caps)
            proxy_mask = sel.get("proxy_slot", pd.Series(False, index=sel.index)).astype(bool)
            proxy_cap = float(spec.proxy_weight_cap or 1.0)
            if proxy_mask.any() and proxy_cap < 0.999999:
                proxy_sum = float(w.reindex(proxy_mask[proxy_mask].index).sum())
                if proxy_sum > proxy_cap + 1e-9:
                    scale = proxy_cap / max(proxy_sum, 1e-12)
                    proxy_idx = proxy_mask[proxy_mask].index
                    before = w.loc[proxy_idx].copy()
                    w.loc[proxy_idx] = w.loc[proxy_idx] * scale
                    excess = float(before.sum() - w.loc[proxy_idx].sum())
                    free_idx = proxy_mask[~proxy_mask].index
                    if len(free_idx) > 0 and excess > 1e-12:
                        free_w = w.reindex(free_idx).clip(lower=0.0)
                        if float(free_w.sum()) <= 0:
                            w.loc[free_idx] = excess / len(free_idx)
                        else:
                            w.loc[free_idx] = w.loc[free_idx] + excess * (free_w / float(free_w.sum()))
                    w = w / max(float(w.sum()), 1e-12)

        # Remove dust allocations while keeping at least min_contracts.
        if len(w) > spec.min_contracts:
            min_contract_weight = 0.01
            keep = w[w >= min_contract_weight]
            if len(keep) >= spec.min_contracts:
                w = keep
                selected = selected[selected["market_id"].isin(w.index)].copy()
                w = _clip_and_normalize(w, max_weight=spec.max_single_weight)

        investable = max(0.0, 1.0 - spec.cash_weight)
        w = w * investable

        out = selected.copy()
        out["target_weight"] = out["market_id"].map(w).fillna(0.0)
        out["proxy_weight_cap"] = float(spec.proxy_weight_cap)
        return out

    def build_for_date(self, rebalance_date: str | pd.Timestamp) -> pd.DataFrame:
        as_of = pd.Timestamp(rebalance_date)
        rows: list[pd.DataFrame] = []

        for spec in self.specs:
            candidates = self._filter_theme_candidates(as_of, spec)
            selected = self._select_contracts(candidates, spec)
            weighted = self._weight_selected(selected, spec)

            if not weighted.empty:
                weighted = weighted.copy()
                weighted["spot_effective_price_raw"] = weighted["effective_risk_price"]
                weighted["spot_effective_price_horizon"] = weighted.apply(
                    lambda row: _horizon_normalized_probability(
                        row.get("effective_risk_price"),
                        row.get("days_to_expiry"),
                        row.get("spot_horizon_days", spec.spot_horizon_days),
                    ),
                    axis=1,
                )
                weighted["rebalance_date"] = as_of
                weighted["domain"] = spec.domain
                weighted["basket_code"] = spec.basket_code
                weighted["basket_name"] = spec.basket_name
                weighted["basket_weight"] = spec.basket_weight
                weighted["cash_weight"] = spec.cash_weight
                weighted["portfolio_weight"] = weighted["target_weight"] * spec.basket_weight
                weighted["is_cash"] = False
                weighted["turnover"] = 0.0
                weighted["treasury_risk"] = 0.0
                weighted["broad_risk"] = 0.0
                rows.append(
                    weighted[
                        [
                            "rebalance_date",
                            "domain",
                            "basket_code",
                            "basket_name",
                            "basket_weight",
                            "cash_weight",
                            "market_id",
                            "ticker_id",
                            "ticker_name",
                            "title",
                            "platform",
                            "event_slug",
                            "event_family_key",
                            "exclusive_group_key",
                            "llm_category",
                            "llm_secondary_category",
                            "classification_source",
                            "classification_confidence",
                            "exposure_direction",
                            "exposure_confidence",
                            "position_side",
                            "position_instruction",
                            "side_source",
                            "direction_reason",
                            "current_price",
                            "market_yes_price",
                            "effective_price",
                            "effective_risk_price",
                            "spot_effective_price_raw",
                            "spot_effective_price_horizon",
                            "listed_at",
                            "inactive_at",
                            "temporal_source",
                            "first_seen_proxy",
                            "end_date",
                            "days_to_expiry",
                            "tenor_target_days",
                            "tenor_min_days",
                            "tenor_max_days",
                            "roll_floor_days",
                            "spot_horizon_days",
                            "tenor_distance_days",
                            "tenor_band_status",
                            "slot_key",
                            "slot_name",
                            "proxy_slot",
                            "slot_selection_status",
                            "proxy_weight_cap",
                            "keyword_score",
                            "direct_score",
                            "inverse_score",
                            "theme_side_score",
                            "preferred_probability_band",
                            "quality_score",
                            "selection_score",
                            "target_weight",
                            "portfolio_weight",
                            "is_cash",
                            "turnover",
                            "treasury_risk",
                            "broad_risk",
                        ]
                    ]
                )

            cash_row = pd.DataFrame(
                [
                    {
                        "rebalance_date": as_of,
                        "domain": spec.domain,
                        "basket_code": spec.basket_code,
                        "basket_name": spec.basket_name,
                        "basket_weight": spec.basket_weight,
                        "cash_weight": spec.cash_weight,
                        "market_id": "__CASH__",
                        "ticker_id": "__CASH__",
                        "ticker_name": "Cash Buffer",
                        "title": "Cash Buffer",
                        "platform": "cash",
                        "event_slug": "__CASH__",
                        "event_family_key": "__CASH__",
                        "exclusive_group_key": "__CASH__",
                        "llm_category": "cash",
                        "llm_secondary_category": "",
                        "classification_source": "cash",
                        "classification_confidence": 1.0,
                        "exposure_direction": "",
                        "exposure_confidence": 0.0,
                        "position_side": "CASH",
                        "position_instruction": "CASH",
                        "side_source": "cash",
                        "direction_reason": "cash_buffer",
                        "current_price": np.nan,
                        "market_yes_price": np.nan,
                        "effective_price": 1.0,
                        "effective_risk_price": 1.0,
                        "spot_effective_price_raw": 1.0,
                        "spot_effective_price_horizon": 1.0,
                        "listed_at": pd.NaT,
                        "inactive_at": pd.NaT,
                        "temporal_source": "cash",
                        "first_seen_proxy": pd.NaT,
                        "end_date": pd.NaT,
                        "days_to_expiry": np.nan,
                        "tenor_target_days": int(spec.tenor_target_days),
                        "tenor_min_days": int(spec.tenor_min_days),
                        "tenor_max_days": int(spec.tenor_max_days),
                        "roll_floor_days": int(spec.roll_floor_days),
                        "spot_horizon_days": int(spec.spot_horizon_days),
                        "tenor_distance_days": np.nan,
                        "tenor_band_status": "fallback",
                        "slot_key": "__CASH__",
                        "slot_name": "Cash Buffer",
                        "proxy_slot": False,
                        "slot_selection_status": "cash",
                        "proxy_weight_cap": float(spec.proxy_weight_cap),
                        "keyword_score": 0.0,
                        "direct_score": 0.0,
                        "inverse_score": 0.0,
                        "theme_side_score": 0.0,
                        "preferred_probability_band": "cash",
                        "quality_score": 1.0,
                        "selection_score": 0.0,
                        "target_weight": spec.cash_weight,
                        "portfolio_weight": spec.cash_weight * spec.basket_weight,
                        "is_cash": True,
                        "turnover": 0.0,
                        "treasury_risk": 0.0,
                        "broad_risk": 0.0,
                    }
                ]
            )
            rows.append(cash_row)

        if not rows:
            return pd.DataFrame()

        out = pd.concat(rows, ignore_index=True)
        out = out.sort_values(
            ["rebalance_date", "domain", "basket_code", "is_cash", "target_weight"],
            ascending=[True, True, True, True, False],
        ).reset_index(drop=True)
        return out

    def build_for_dates(self, rebalance_dates: Iterable[pd.Timestamp], *, log_progress: bool = False) -> pd.DataFrame:
        blocks: list[pd.DataFrame] = []
        prev_weights_by_basket: dict[str, pd.Series] = {}
        slot_diag_rows: list[dict] = []

        factor_treasury_cols = ["beta_TNX", "beta_TLT", "beta_SHY", "beta_TLH", "beta_TYX", "beta_FVX", "beta_IRX"]
        factor_broad_cols = ["beta_SPY", "beta_QQQ", "beta_VIX", "beta_USO", "beta_GLD", "beta_BTC_USD"]

        dates = list(pd.to_datetime(list(rebalance_dates)))
        total_dates = len(dates)
        total_specs = len(self.specs)
        total_steps = max(total_dates * max(total_specs, 1), 1)
        step = 0

        for date_idx, d in enumerate(dates, start=1):
            as_of = pd.Timestamp(d)
            if log_progress:
                pct = 100.0 * float(date_idx - 1) / float(max(total_dates, 1))
                print(f"[build] {pct:5.1f}% starting {as_of.date().isoformat()} ({date_idx}/{total_dates})", flush=True)

            for spec_idx, spec in enumerate(self.specs, start=1):
                candidates = self._filter_theme_candidates(as_of, spec)
                selected = self._select_contracts(candidates, spec)
                weighted = self._weight_selected(selected, spec)
                step += 1
                if log_progress:
                    progress_pct = 100.0 * float(step) / float(total_steps)
                    selected_n = 0 if weighted.empty else int(len(weighted))
                    print(
                        f"[build] {progress_pct:5.1f}% {as_of.date().isoformat()} {spec.basket_code} "
                        f"candidates={int(len(candidates))} selected={selected_n} ({spec_idx}/{total_specs})",
                        flush=True,
                    )

                if spec.slot_schema:
                    for slot in spec.slot_schema:
                        slot_candidates = candidates[candidates.get("slot_key", pd.Series("", index=candidates.index)).astype(str) == str(slot.slot_key)].copy()
                        slot_selected = weighted[weighted.get("slot_key", pd.Series("", index=weighted.index)).astype(str) == str(slot.slot_key)].copy() if not weighted.empty else pd.DataFrame()
                        fallback_reason = ""
                        if slot_candidates.empty:
                            slot_status = "no_candidates"
                            if slot.required_if_available:
                                fallback_reason = "no_matching_candidates"
                        elif slot_selected.empty:
                            slot_status = "candidate_unselected"
                            if slot.required_if_available:
                                fallback_reason = "required_slot_unfilled"
                        elif (slot_selected.get("preferred_probability_band", pd.Series("", index=slot_selected.index)) == "tail").all():
                            slot_status = "tail_only"
                        elif (slot_selected.get("preferred_probability_band", pd.Series("", index=slot_selected.index)) == "soft").all():
                            slot_status = "soft_only"
                        else:
                            slot_status = "selected"
                        slot_diag_rows.append(
                            {
                                "rebalance_date": pd.Timestamp(as_of).date().isoformat(),
                                "basket_code": spec.basket_code,
                                "slot_key": slot.slot_key,
                                "slot_name": slot.slot_name,
                                "candidate_count": int(len(slot_candidates)),
                                "selected_count": int(len(slot_selected)),
                                "selected_weight": float(slot_selected.get("target_weight", pd.Series(dtype=float)).sum()) if not slot_selected.empty else 0.0,
                                "required_if_available": bool(slot.required_if_available),
                                "slot_status": slot_status,
                                "fallback_reason": fallback_reason,
                            }
                        )

                if weighted.empty:
                    cash_only = pd.DataFrame(
                        [
                            {
                                "rebalance_date": as_of,
                                "domain": spec.domain,
                                "basket_code": spec.basket_code,
                                "basket_name": spec.basket_name,
                                "basket_weight": spec.basket_weight,
                                "cash_weight": 1.0,
                                "market_id": "__CASH__",
                                "ticker_id": "__CASH__",
                                "ticker_name": "Cash Buffer",
                                "title": "Cash Buffer",
                                "platform": "cash",
                                "event_slug": "__CASH__",
                                "event_family_key": "__CASH__",
                                "exclusive_group_key": "__CASH__",
                                "llm_category": "cash",
                                "llm_secondary_category": "",
                                "classification_source": "cash",
                                "classification_confidence": 1.0,
                                "exposure_direction": "",
                                "exposure_confidence": 0.0,
                                "position_side": "CASH",
                                "position_instruction": "CASH",
                                "side_source": "cash",
                                "direction_reason": "cash_buffer",
                                "current_price": np.nan,
                                "market_yes_price": np.nan,
                                "effective_price": 1.0,
                                "effective_risk_price": 1.0,
                                "spot_effective_price_raw": 1.0,
                                "spot_effective_price_horizon": 1.0,
                                "listed_at": pd.NaT,
                                "inactive_at": pd.NaT,
                                "temporal_source": "cash",
                                "first_seen_proxy": pd.NaT,
                                "end_date": pd.NaT,
                                "days_to_expiry": np.nan,
                                "tenor_target_days": int(spec.tenor_target_days),
                                "tenor_min_days": int(spec.tenor_min_days),
                                "tenor_max_days": int(spec.tenor_max_days),
                                "roll_floor_days": int(spec.roll_floor_days),
                                "spot_horizon_days": int(spec.spot_horizon_days),
                                "tenor_distance_days": np.nan,
                                "tenor_band_status": "fallback",
                                "slot_key": "__CASH__",
                                "slot_name": "Cash Buffer",
                                "proxy_slot": False,
                                "slot_selection_status": "cash",
                                "proxy_weight_cap": float(spec.proxy_weight_cap),
                                "keyword_score": 0.0,
                                "direct_score": 0.0,
                                "inverse_score": 0.0,
                                "theme_side_score": 0.0,
                                "preferred_probability_band": "cash",
                                "quality_score": 0.0,
                                "selection_score": 0.0,
                                "target_weight": 1.0,
                                "portfolio_weight": spec.basket_weight,
                                "is_cash": True,
                                "turnover": 0.0,
                                "treasury_risk": 0.0,
                                "broad_risk": 0.0,
                            }
                        ]
                    )
                    blocks.append(cash_only)
                    prev_weights_by_basket[spec.basket_code] = pd.Series(dtype=float)
                    continue

                # Compute preliminary non-cash weights.
                w_pre = weighted.set_index("market_id")["target_weight"].astype(float)
                prev_w = prev_weights_by_basket.get(spec.basket_code, pd.Series(dtype=float))
                union = sorted(set(w_pre.index) | set(prev_w.index))
                if union:
                    turnover = float((w_pre.reindex(union, fill_value=0.0) - prev_w.reindex(union, fill_value=0.0)).abs().sum() / 2.0)
                else:
                    turnover = 0.0

                tre_risk = 0.0
                broad_risk = 0.0
                if not weighted.empty:
                    for col in factor_treasury_cols:
                        if col in weighted.columns:
                            tre_risk += abs(float((weighted[col] * weighted["target_weight"]).sum()))
                    for col in factor_broad_cols:
                        if col in weighted.columns:
                            broad_risk += abs(float((weighted[col] * weighted["target_weight"]).sum()))

                # Dynamic cash sleeve: base + turnover/risk stress; clipped.
                scarcity = max(0.0, spec.min_contracts - len(weighted)) / max(spec.min_contracts, 1)
                cash_dynamic = spec.cash_weight + min(0.08, 0.22 * turnover) + min(0.08, 0.55 * tre_risk) + min(0.04, 0.20 * broad_risk) + min(0.05, 0.20 * scarcity)
                cash_dynamic = float(min(0.35, max(spec.cash_weight, cash_dynamic)))
                investable = max(0.0, 1.0 - cash_dynamic)

                sum_pre = float(weighted["target_weight"].sum())
                if sum_pre > 0:
                    weighted["target_weight"] = weighted["target_weight"] * (investable / sum_pre)
                else:
                    weighted["target_weight"] = investable / max(len(weighted), 1)

                weighted = weighted.copy()
                weighted["spot_effective_price_raw"] = weighted["effective_risk_price"]
                weighted["spot_effective_price_horizon"] = weighted.apply(
                    lambda row: _horizon_normalized_probability(
                        row.get("effective_risk_price"),
                        row.get("days_to_expiry"),
                        row.get("spot_horizon_days", spec.spot_horizon_days),
                    ),
                    axis=1,
                )
                weighted["rebalance_date"] = as_of
                weighted["domain"] = spec.domain
                weighted["basket_code"] = spec.basket_code
                weighted["basket_name"] = spec.basket_name
                weighted["basket_weight"] = spec.basket_weight
                weighted["cash_weight"] = cash_dynamic
                weighted["portfolio_weight"] = weighted["target_weight"] * spec.basket_weight
                weighted["is_cash"] = False
                weighted["turnover"] = turnover
                weighted["treasury_risk"] = tre_risk
                weighted["broad_risk"] = broad_risk

                factor_cols_keep = [col for col in weighted.columns if col.startswith("beta_")]
                keep_cols = [
                    "rebalance_date",
                    "domain",
                    "basket_code",
                    "basket_name",
                    "basket_weight",
                    "cash_weight",
                    "market_id",
                    "ticker_id",
                    "ticker_name",
                    "title",
                    "platform",
                    "event_slug",
                    "event_family_key",
                    "exclusive_group_key",
                    "llm_category",
                    "llm_secondary_category",
                    "classification_source",
                    "classification_confidence",
                    "exposure_direction",
                    "exposure_confidence",
                    "position_side",
                    "position_instruction",
                    "side_source",
                    "direction_reason",
                    "current_price",
                    "market_yes_price",
                    "effective_price",
                    "effective_risk_price",
                    "spot_effective_price_raw",
                    "spot_effective_price_horizon",
                    "listed_at",
                    "inactive_at",
                    "temporal_source",
                    "first_seen_proxy",
                    "end_date",
                    "days_to_expiry",
                    "tenor_target_days",
                    "tenor_min_days",
                    "tenor_max_days",
                    "roll_floor_days",
                    "spot_horizon_days",
                    "tenor_distance_days",
                    "tenor_band_status",
                    "slot_key",
                    "slot_name",
                    "proxy_slot",
                    "slot_selection_status",
                    "proxy_weight_cap",
                    "keyword_score",
                    "direct_score",
                    "inverse_score",
                    "theme_side_score",
                    "preferred_probability_band",
                    "quality_score",
                    "selection_score",
                    "target_weight",
                    "portfolio_weight",
                    "is_cash",
                    "turnover",
                    "treasury_risk",
                    "broad_risk",
                ]
                keep_cols.extend(factor_cols_keep)
                blocks.append(weighted[keep_cols])

                cash_row = pd.DataFrame(
                    [
                        {
                            "rebalance_date": as_of,
                            "domain": spec.domain,
                            "basket_code": spec.basket_code,
                            "basket_name": spec.basket_name,
                            "basket_weight": spec.basket_weight,
                            "cash_weight": cash_dynamic,
                            "market_id": "__CASH__",
                            "ticker_id": "__CASH__",
                            "ticker_name": "Cash Buffer",
                            "title": "Cash Buffer",
                            "platform": "cash",
                            "event_slug": "__CASH__",
                            "event_family_key": "__CASH__",
                            "exclusive_group_key": "__CASH__",
                            "llm_category": "cash",
                            "llm_secondary_category": "",
                            "classification_source": "cash",
                            "classification_confidence": 1.0,
                            "exposure_direction": "",
                            "exposure_confidence": 0.0,
                            "position_side": "CASH",
                            "position_instruction": "CASH",
                            "side_source": "cash",
                            "direction_reason": "cash_buffer",
                            "current_price": np.nan,
                            "market_yes_price": np.nan,
                            "effective_price": 1.0,
                            "effective_risk_price": 1.0,
                            "spot_effective_price_raw": 1.0,
                            "spot_effective_price_horizon": 1.0,
                            "listed_at": pd.NaT,
                            "inactive_at": pd.NaT,
                            "temporal_source": "cash",
                            "first_seen_proxy": pd.NaT,
                            "end_date": pd.NaT,
                            "days_to_expiry": np.nan,
                            "tenor_target_days": int(spec.tenor_target_days),
                            "tenor_min_days": int(spec.tenor_min_days),
                            "tenor_max_days": int(spec.tenor_max_days),
                            "roll_floor_days": int(spec.roll_floor_days),
                            "spot_horizon_days": int(spec.spot_horizon_days),
                            "tenor_distance_days": np.nan,
                            "tenor_band_status": "fallback",
                            "slot_key": "__CASH__",
                            "slot_name": "Cash Buffer",
                            "proxy_slot": False,
                            "slot_selection_status": "cash",
                            "proxy_weight_cap": float(spec.proxy_weight_cap),
                            "keyword_score": 0.0,
                            "direct_score": 0.0,
                            "inverse_score": 0.0,
                            "theme_side_score": 0.0,
                            "preferred_probability_band": "cash",
                            "quality_score": 1.0,
                            "selection_score": 0.0,
                            "target_weight": cash_dynamic,
                            "portfolio_weight": cash_dynamic * spec.basket_weight,
                            "is_cash": True,
                            "turnover": turnover,
                            "treasury_risk": tre_risk,
                            "broad_risk": broad_risk,
                        }
                    ]
                )
                blocks.append(cash_row)

                prev_weights_by_basket[spec.basket_code] = weighted.set_index("market_id")["target_weight"].astype(float)

        if not blocks:
            self.last_slot_coverage_diagnostics = pd.DataFrame(slot_diag_rows)
            return pd.DataFrame()
        out = pd.concat(blocks, ignore_index=True)
        out = out.sort_values(["rebalance_date", "domain", "basket_code", "is_cash", "target_weight"], ascending=[True, True, True, True, False])
        self.last_slot_coverage_diagnostics = pd.DataFrame(slot_diag_rows).sort_values(["rebalance_date", "basket_code", "slot_key"]).reset_index(drop=True) if slot_diag_rows else pd.DataFrame()
        return out.reset_index(drop=True)


def summarize_latest_baskets(compositions: pd.DataFrame) -> pd.DataFrame:
    if compositions.empty:
        return pd.DataFrame()
    c = compositions.copy()
    c["rebalance_date"] = pd.to_datetime(c["rebalance_date"])
    latest = pd.Timestamp(c["rebalance_date"].max())
    cur = c[c["rebalance_date"] == latest].copy()
    non_cash = cur[~cur["is_cash"]].copy()

    if non_cash.empty:
        return pd.DataFrame()

    summary = (
        non_cash.groupby(["domain", "basket_code", "basket_name"], as_index=False)
        .agg(
            basket_weight=("basket_weight", "first"),
            cash_weight=("cash_weight", "first"),
            n_contracts=("market_id", "nunique"),
            avg_contract_weight=("target_weight", "mean"),
            max_contract_weight=("target_weight", "max"),
            avg_days_to_expiry=("days_to_expiry", "mean"),
        )
        .sort_values(["domain", "basket_code"])
        .reset_index(drop=True)
    )
    summary["as_of"] = latest.date().isoformat()
    return summary


def _write_markdown_composition(compositions: pd.DataFrame, output_path: Path) -> None:
    if compositions.empty:
        output_path.write_text("# Thematic Basket Compositions\n\nNo rows generated.\n", encoding="utf-8")
        return

    c = compositions.copy()
    c["rebalance_date"] = pd.to_datetime(c["rebalance_date"])
    c["end_date"] = pd.to_datetime(c["end_date"], errors="coerce")
    c = c.sort_values(["basket_code", "rebalance_date", "is_cash", "target_weight"], ascending=[True, True, True, False])

    lines = [
        "# Thematic Basket Monthly Composition (Last Year)",
        "",
        "Rules enforced: no election contracts, 10-50 contracts per basket, explicit cash sleeve.",
        "",
    ]

    for basket_code, grp in c.groupby("basket_code", sort=True):
        basket_name = str(grp["basket_name"].iloc[0])
        domain = str(grp["domain"].iloc[0])
        lines.append(f"## {domain} | {basket_code} | {basket_name}")
        lines.append("")
        for d, g in grp.groupby("rebalance_date", sort=True):
            non_cash = g[~g["is_cash"]].copy()
            cash = g[g["is_cash"]].copy()
            cash_w = float(cash["target_weight"].iloc[0]) if not cash.empty else 0.0
            lines.append(f"### {pd.Timestamp(d).date().isoformat()} (contracts={len(non_cash)}, cash={cash_w:.2%})")
            lines.append("")
            lines.append("| Weight | Market ID | Contract | Expiry |")
            lines.append("| ---: | --- | --- | --- |")
            for _, row in non_cash.head(50).iterrows():
                expiry = "—" if pd.isna(row["end_date"]) else pd.Timestamp(row["end_date"]).date().isoformat()
                lines.append(
                    f"| {float(row['target_weight']):.4f} | {row['market_id']} | {row['title']} | {expiry} |"
                )
            lines.append("")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_latest_markdown(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        output_path.write_text("# Final Thematic Basket List\n\nNo rows generated.\n", encoding="utf-8")
        return
    s = summary.copy()
    lines = [
        "# Final Thematic Basket List",
        "",
        "| Domain | Code | Basket | Basket Weight | Cash | Contracts | Avg Contract Weight | Avg DTE | As Of |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, r in s.iterrows():
        lines.append(
            "| "
            + f"{r['domain']} | {r['basket_code']} | {r['basket_name']} | {float(r['basket_weight']):.4f} | "
            + f"{float(r['cash_weight']):.2%} | {int(r['n_contracts'])} | {float(r['avg_contract_weight']):.4f} | "
            + f"{float(r['avg_days_to_expiry']):.1f} | {r['as_of']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latest_html(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        output_path.write_text("<html><body><p>No basket rows.</p></body></html>", encoding="utf-8")
        return

    view = summary.copy()
    view["basket_weight"] = view["basket_weight"].map(lambda x: f"{float(x):.4f}")
    view["cash_weight"] = view["cash_weight"].map(lambda x: f"{float(x):.2%}")
    view["avg_contract_weight"] = view["avg_contract_weight"].map(lambda x: f"{float(x):.4f}")
    view["max_contract_weight"] = view["max_contract_weight"].map(lambda x: f"{float(x):.4f}")
    view["avg_days_to_expiry"] = view["avg_days_to_expiry"].map(lambda x: f"{float(x):.1f}")

    col_map = {
        "domain": "Domain",
        "basket_code": "Code",
        "basket_name": "Basket",
        "basket_weight": "Basket Weight",
        "cash_weight": "Cash",
        "n_contracts": "Contracts",
        "avg_contract_weight": "Avg Contract Weight",
        "max_contract_weight": "Max Contract Weight",
        "avg_days_to_expiry": "Avg DTE",
        "as_of": "As Of",
    }
    view = view[[c for c in col_map if c in view.columns]].rename(columns=col_map)
    table_html = view.to_html(index=False, classes=["basket-table"])

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Final Thematic Basket List</title>
  <style>
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
      background: #f7f8fb;
      color: #111827;
    }}
    .wrap {{
      width: min(1700px, 98vw);
      margin: 16px auto;
      background: #ffffff;
      border: 1px solid #d1d5db;
      border-radius: 10px;
      overflow: auto;
    }}
    h1 {{
      margin: 0;
      padding: 14px 16px 8px;
      font-size: 20px;
    }}
    p {{
      margin: 0;
      padding: 0 16px 12px;
      color: #6b7280;
      font-size: 13px;
    }}
    table.basket-table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
    }}
    table.basket-table th, table.basket-table td {{
      border-top: 1px solid #e5e7eb;
      padding: 7px 8px;
      text-align: left;
      word-break: break-word;
    }}
    table.basket-table th {{
      background: #f3f4f6;
      position: sticky;
      top: 0;
    }}
    table.basket-table th:nth-child(4), table.basket-table td:nth-child(4),
    table.basket-table th:nth-child(5), table.basket-table td:nth-child(5),
    table.basket-table th:nth-child(6), table.basket-table td:nth-child(6),
    table.basket-table th:nth-child(7), table.basket-table td:nth-child(7),
    table.basket-table th:nth-child(8), table.basket-table td:nth-child(8),
    table.basket-table th:nth-child(9), table.basket-table td:nth-child(9) {{
      text-align: right;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Final Thematic Basket List</h1>
    <p>Monthly thematic system (no election baskets, 10-50 contracts, explicit cash sleeve).</p>
    {table_html}
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _slug(text: str, max_len: int = 120) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text).strip()).strip("-")
    if not s:
        s = "item"
    return s[:max_len]


def _safe_delete(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _render_basket_composition_chart(basket_non_cash: pd.DataFrame, basket_name: str, chart_path: Path) -> None:
    plt = _get_plt()
    pivot = (
        basket_non_cash.pivot(index="rebalance_date", columns="market_id", values="target_weight")
        .fillna(0.0)
        .sort_index()
    )
    if pivot.empty:
        return

    x = pd.to_datetime(pivot.index)
    y = pivot.to_numpy().T

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(x, y, alpha=0.92)
    ax.axhline(0.90, color="#374151", linestyle="--", linewidth=1.0, alpha=0.8, label="Invested Sleeve (90%)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{basket_name} - Contract Composition Over Time")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Rebalance Date")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def _render_contract_weight_chart(contract_df: pd.DataFrame, chart_path: Path, title: str) -> None:
    plt = _get_plt()
    s = (
        contract_df.set_index("rebalance_date")["target_weight"]
        .sort_index()
        .astype(float)
    )
    fig, ax = plt.subplots(figsize=(10.5, 3.0))
    ax.plot(s.index, s.values, color="#1d4ed8", linewidth=1.8, marker="o", markersize=3.5)
    ax.set_ylim(0.0, max(0.12, float(s.max()) * 1.2))
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Weight")
    ax.set_xlabel("Rebalance Date")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=140)
    plt.close(fig)


def _build_basket_level_series(
    compositions: pd.DataFrame,
    transitions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if compositions is None or compositions.empty:
        return pd.DataFrame()

    c = compositions.copy()
    c["rebalance_date"] = pd.to_datetime(c["rebalance_date"], errors="coerce")
    c["target_weight"] = pd.to_numeric(c["target_weight"], errors="coerce").fillna(0.0)
    c["basket_weight"] = pd.to_numeric(c["basket_weight"], errors="coerce").fillna(0.0)
    c["current_price"] = pd.to_numeric(c.get("current_price", np.nan), errors="coerce")
    c["effective_price"] = pd.to_numeric(c.get("effective_price", np.nan), errors="coerce")
    non_cash = c[~c["is_cash"]].copy()
    cash = c[c["is_cash"]][["rebalance_date", "basket_code", "target_weight"]].rename(columns={"target_weight": "cash_weight"})

    if non_cash.empty:
        return pd.DataFrame()

    basket_rows: list[dict] = []
    for keys, g in non_cash.groupby(["rebalance_date", "domain", "basket_code", "basket_name"], sort=True):
        d, domain, code, basket_name = keys
        bw = float(g["basket_weight"].iloc[0]) if len(g) else 0.0
        px = g["effective_price"].copy()
        if px.notna().sum() == 0:
            px = g["current_price"].copy()
        px_cov = float(px.notna().mean()) if len(px) else 0.0
        px = px.fillna(0.5).clip(lower=0.0, upper=1.0)
        weighted_px = float((g["target_weight"] * px).sum())
        cw = float(cash[(cash["rebalance_date"] == d) & (cash["basket_code"] == code)]["cash_weight"].iloc[0]) if not cash.empty and ((cash["rebalance_date"] == d) & (cash["basket_code"] == code)).any() else max(0.0, 1.0 - float(g["target_weight"].sum()))
        basket_level = 100.0 * (weighted_px + cw)
        basket_rows.append(
            {
                "rebalance_date": pd.Timestamp(d).date().isoformat(),
                "domain": str(domain),
                "basket_code": str(code),
                "basket_name": str(basket_name),
                "basket_weight": bw,
                "basket_level": basket_level,
                "cash_weight": cw,
                "price_coverage_share": px_cov,
            }
        )

    out = pd.DataFrame(basket_rows)
    if out.empty:
        return out

    if isinstance(transitions, pd.DataFrame) and not transitions.empty:
        tr = transitions.copy()
        tr["rebalance_date"] = pd.to_datetime(tr["rebalance_date"], errors="coerce").dt.date.astype(str)
        tr_b = (
            tr[["rebalance_date", "basket_code", "entries_n", "exits_n", "turnover"]]
            .rename(columns={"entries_n": "entries", "exits_n": "exits"})
            .copy()
        )
        out = out.merge(tr_b, on=["rebalance_date", "basket_code"], how="left")
    if "entries" not in out.columns:
        out["entries"] = 0
    if "exits" not in out.columns:
        out["exits"] = 0
    if "turnover" not in out.columns:
        out["turnover"] = 0.0
    out["entries"] = pd.to_numeric(out["entries"], errors="coerce").fillna(0).astype(int)
    out["exits"] = pd.to_numeric(out["exits"], errors="coerce").fillna(0).astype(int)
    out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce").fillna(0.0)
    return out.sort_values(["rebalance_date", "basket_code"]).reset_index(drop=True)


def _build_aggregate_basket_series(
    compositions: pd.DataFrame,
    transitions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    br = _build_basket_level_series(compositions, transitions)
    if br.empty:
        return br

    agg_rows: list[dict] = []
    for d, g in br.groupby("rebalance_date", sort=True):
        w = g["basket_weight"].astype(float)
        wsum = float(w.sum())
        if wsum <= 0:
            wn = pd.Series(np.ones(len(g)) / max(len(g), 1), index=g.index)
        else:
            wn = w / wsum
        agg_rows.append(
            {
                "rebalance_date": str(d),
                "overall_basket_level": float((g["basket_level"] * wn).sum()),
                "overall_cash_weight": float((g["cash_weight"] * wn).sum()),
                "overall_price_coverage_share": float((g["price_coverage_share"] * wn).sum()),
                "n_baskets": int(g["basket_code"].nunique()),
                "total_entries": int(g["entries"].sum()),
                "total_exits": int(g["exits"].sum()),
                "total_turnover": float(g["turnover"].sum()),
            }
        )
    out = pd.DataFrame(agg_rows).sort_values("rebalance_date")
    return out


def _render_aggregate_level_chart(aggregate: pd.DataFrame, chart_path: Path) -> None:
    plt = _get_plt()
    if aggregate is None or aggregate.empty:
        return

    d = aggregate.copy()
    d["rebalance_date"] = pd.to_datetime(d["rebalance_date"], errors="coerce")
    d = d.sort_values("rebalance_date")
    if d.empty:
        return

    fig, ax1 = plt.subplots(figsize=(12.0, 4.0))
    ax1.plot(d["rebalance_date"], d["overall_basket_level"], color="#1d4ed8", linewidth=2.1, marker="o", markersize=4, label="Overall Basket Level")
    for dt in d["rebalance_date"]:
        ax1.axvline(dt, color="#cbd5e1", linewidth=0.7, linestyle="--", alpha=0.55)
    ax1.set_ylabel("Level (0-100 proxy)")
    ax1.set_xlabel("Rebalance Date")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    changes = d["total_entries"].astype(float) + d["total_exits"].astype(float)
    ax2.bar(d["rebalance_date"], changes, width=18, color="#94a3b8", alpha=0.35, label="Entries+Exits")
    ax2.set_ylabel("Composition Changes")

    ax1.set_title("Overall Basket Level With Rebalance Markers")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def _render_aggregate_cash_chart(aggregate: pd.DataFrame, chart_path: Path) -> None:
    plt = _get_plt()
    if aggregate is None or aggregate.empty:
        return
    d = aggregate.copy()
    d["rebalance_date"] = pd.to_datetime(d["rebalance_date"], errors="coerce")
    d = d.sort_values("rebalance_date")
    if d.empty:
        return

    fig, ax1 = plt.subplots(figsize=(12.0, 3.8))
    ax1.plot(d["rebalance_date"], d["overall_cash_weight"], color="#059669", linewidth=2.0, marker="o", markersize=3.5)
    ax1.set_ylabel("Weighted Cash Weight")
    ax1.set_xlabel("Rebalance Date")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(d["rebalance_date"], d["total_turnover"], color="#7c3aed", linewidth=1.8, marker="s", markersize=3, alpha=0.9)
    ax2.set_ylabel("Total Turnover")

    ax1.set_title("Aggregate Cash And Turnover Through Time")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def _table_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "<p class='muted'>No rows available.</p>"
    return df.head(max_rows).to_html(index=False, classes=["tbl"], border=0, escape=False)


def _ensure_inception_override_file(
    specs: list[ThematicBasketSpec],
    path: Path = INCEPTION_OVERRIDE_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    default_doc = {spec.basket_code: {"date": None, "reason": None} for spec in specs}
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(default_doc, f, sort_keys=True)
        return path

    try:
        with open(path, "r", encoding="utf-8") as f:
            current = yaml.safe_load(f) or {}
    except Exception:
        current = {}

    changed = False
    for code, payload in default_doc.items():
        if code not in current or not isinstance(current.get(code), dict):
            current[code] = payload
            changed = True
        else:
            current[code].setdefault("date", None)
            current[code].setdefault("reason", None)
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(current, f, sort_keys=True)
    return path


def _load_inception_overrides(
    specs: list[ThematicBasketSpec],
    path: Path = INCEPTION_OVERRIDE_PATH,
) -> dict[str, dict[str, str | None]]:
    _ensure_inception_override_file(specs, path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        raw = {}

    out: dict[str, dict[str, str | None]] = {}
    for spec in specs:
        payload = raw.get(spec.basket_code, {}) if isinstance(raw, dict) else {}
        out[spec.basket_code] = {
            "date": None if not isinstance(payload, dict) else (str(payload.get("date")) if payload.get("date") else None),
            "reason": None if not isinstance(payload, dict) else (str(payload.get("reason")) if payload.get("reason") else None),
        }
    return out


def _build_inception_policy(
    basket_level_series: pd.DataFrame,
    specs: list[ThematicBasketSpec],
    overrides: dict[str, dict[str, str | None]] | None = None,
) -> pd.DataFrame:
    codes = [spec.basket_code for spec in specs]
    names = {spec.basket_code: spec.basket_name for spec in specs}
    bl = basket_level_series.copy() if isinstance(basket_level_series, pd.DataFrame) else pd.DataFrame()
    if bl.empty:
        return pd.DataFrame(
            [
                {
                    "basket_code": code,
                    "basket_name": names.get(code, code),
                    "raw_history_start": "",
                    "default_inception_date": "",
                    "manual_override_date": (overrides or {}).get(code, {}).get("date") or "",
                    "effective_inception_date": "",
                    "price_coverage_at_inception": np.nan,
                    "cash_weight_at_inception": np.nan,
                    "rule_name": "no_history",
                    "override_reason": (overrides or {}).get(code, {}).get("reason") or "",
                }
                for code in codes
            ]
        )

    bl["rebalance_date"] = pd.to_datetime(bl["rebalance_date"], errors="coerce")
    rows: list[dict] = []
    for code in codes:
        g = bl[bl["basket_code"].astype(str) == str(code)].sort_values("rebalance_date").copy()
        override = (overrides or {}).get(code, {})
        raw_start = pd.Timestamp(g["rebalance_date"].min()).date().isoformat() if not g.empty else ""
        auto = g[(pd.to_numeric(g.get("price_coverage_share"), errors="coerce") >= 0.60) & (pd.to_numeric(g.get("cash_weight"), errors="coerce") <= 0.25)].head(1)
        fallback = g[pd.to_numeric(g.get("cash_weight"), errors="coerce") < 0.95].head(1)

        default_row = auto.iloc[0] if not auto.empty else None
        fallback_row = fallback.iloc[0] if not fallback.empty else None

        manual_date = override.get("date")
        manual_reason = override.get("reason") or ""
        effective_row = None
        rule_name = ""
        if manual_date:
            manual_dt = pd.to_datetime(manual_date, errors="coerce")
            if pd.notna(manual_dt):
                manual_match = g[g["rebalance_date"] >= manual_dt].head(1)
                if not manual_match.empty:
                    effective_row = manual_match.iloc[0]
                    rule_name = "manual_override"
        if effective_row is None and default_row is not None:
            effective_row = default_row
            rule_name = "coverage>=0.60_cash<=0.25"
        if effective_row is None and fallback_row is not None:
            effective_row = fallback_row
            rule_name = "cash<0.95_fallback"
        if effective_row is None and not g.empty:
            effective_row = g.iloc[0]
            rule_name = "raw_history_start_fallback"
        if effective_row is None:
            rule_name = "no_history"

        default_date = pd.Timestamp(default_row["rebalance_date"]).date().isoformat() if default_row is not None else ""
        effective_date = pd.Timestamp(effective_row["rebalance_date"]).date().isoformat() if effective_row is not None else ""
        rows.append(
            {
                "basket_code": code,
                "basket_name": names.get(code, code),
                "raw_history_start": raw_start,
                "default_inception_date": default_date,
                "manual_override_date": manual_date or "",
                "effective_inception_date": effective_date,
                "price_coverage_at_inception": float(pd.to_numeric(effective_row.get("price_coverage_share"), errors="coerce")) if effective_row is not None and pd.notna(pd.to_numeric(effective_row.get("price_coverage_share"), errors="coerce")) else np.nan,
                "cash_weight_at_inception": float(pd.to_numeric(effective_row.get("cash_weight"), errors="coerce")) if effective_row is not None and pd.notna(pd.to_numeric(effective_row.get("cash_weight"), errors="coerce")) else np.nan,
                "rule_name": rule_name,
                "override_reason": manual_reason,
            }
        )

    return pd.DataFrame(rows).sort_values("basket_code").reset_index(drop=True)


def _copy_site_assets(site_dir: Path) -> dict[str, str]:
    assets_dir = site_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    if not ECHARTS_ASSET_SOURCE.exists():
        raise FileNotFoundError(f"Missing bundled ECharts asset: {ECHARTS_ASSET_SOURCE}")
    shutil.copyfile(ECHARTS_ASSET_SOURCE, assets_dir / "echarts.min.js")
    return {"echarts_js": "assets/echarts.min.js"}


def _site_base_css(extra: str = "") -> str:
    extra = extra.replace("{{", "{").replace("}}", "}")
    return f"""
    :root {{
      --bg: #eef1f5;
      --panel: rgba(255, 255, 255, 0.96);
      --panel-strong: #ffffff;
      --panel-soft: #f6f8fb;
      --line: #d7dde6;
      --line-strong: #c6d0dd;
      --text: #142033;
      --muted: #5d6777;
      --muted-soft: #7f8a9c;
      --accent: #1a4faa;
      --accent-strong: #123d85;
      --accent-soft: rgba(26, 79, 170, 0.08);
      --success: #0d8c5f;
      --danger: #b42318;
      --shadow: 0 16px 34px rgba(15, 23, 42, 0.06);
      --radius-lg: 18px;
      --radius-md: 12px;
      --radius-sm: 10px;
    }}
    * {{
      box-sizing: border-box;
    }}
    html {{
      scroll-behavior: smooth;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(255, 255, 255, 0.92) 0%, rgba(255, 255, 255, 0) 28%),
        linear-gradient(180deg, #f6f8fb 0%, var(--bg) 52%, #edf1f6 100%);
      color: var(--text);
    }}
    a {{
      color: var(--accent);
    }}
    .topline {{
      height: 5px;
      background: linear-gradient(90deg, #184b9a 0%, #2a6bd5 40%, #184b9a 100%);
      box-shadow: 0 1px 0 rgba(255, 255, 255, 0.6) inset;
    }}
    .shell {{
      width: min(1380px, 94vw);
      margin: 14px auto 34px;
    }}
    .nav-shell {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 18px;
      padding: 14px 18px;
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      background: rgba(255, 255, 255, 0.82);
      backdrop-filter: blur(12px);
      box-shadow: var(--shadow);
      position: sticky;
      top: 12px;
      z-index: 20;
    }}
    .brand-lockup {{
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }}
    .brand-mark {{
      width: 34px;
      height: 34px;
      border-radius: 11px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      font-weight: 700;
      font-size: 16px;
      background: linear-gradient(135deg, #173f86 0%, #2b6ed8 100%);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.18);
    }}
    .brand-copy {{
      min-width: 0;
    }}
    .brand-copy strong {{
      display: block;
      font-size: 19px;
      letter-spacing: -0.02em;
      line-height: 1.1;
    }}
    .brand-copy span {{
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.11em;
      font-weight: 700;
    }}
    .nav-links {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }}
    .nav-links a {{
      text-decoration: none;
      color: var(--muted);
      font-size: 14px;
      font-weight: 700;
      padding: 9px 12px;
      border-radius: 999px;
      transition: background 120ms ease, color 120ms ease;
    }}
    .nav-links a:hover {{
      background: var(--accent-soft);
      color: var(--accent);
    }}
    .nav-links a.active {{
      color: #fff;
      background: linear-gradient(180deg, #1a4faa 0%, #173f86 100%);
      box-shadow: 0 8px 18px rgba(26, 79, 170, 0.24);
    }}
    .hero {{
      margin-top: 16px;
      padding: 28px 30px;
      border: 1px solid var(--line);
      border-radius: 26px;
      background:
        radial-gradient(circle at 92% 15%, rgba(26, 79, 170, 0.10) 0%, rgba(26, 79, 170, 0) 25%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.94) 0%, rgba(248, 250, 253, 0.96) 100%);
      box-shadow: var(--shadow);
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.9fr);
      gap: 20px;
      align-items: start;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--accent);
      font-weight: 800;
      margin-bottom: 12px;
    }}
    .eyebrow::before {{
      content: "";
      width: 28px;
      height: 1px;
      background: var(--accent);
      opacity: 0.65;
    }}
    h1, h2, h3 {{
      color: var(--text);
      letter-spacing: -0.03em;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(32px, 4vw, 52px);
      line-height: 0.98;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 28px;
      line-height: 1.06;
    }}
    h3 {{
      margin: 0 0 8px;
      font-size: 17px;
      line-height: 1.15;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.62;
      font-size: 14px;
    }}
    .hero-copy p {{
      max-width: 820px;
      font-size: 15px;
    }}
    .hero-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }}
    .hero-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.82);
      color: var(--text);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.03em;
    }}
    .hero-chip strong {{
      font-size: 12px;
    }}
    .hero-profile {{
      display: grid;
      gap: 10px;
    }}
    .profile-card, .panel, .card {{
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .profile-card {{
      padding: 16px 18px;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(247, 249, 252, 0.96) 100%);
    }}
    .profile-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 14px;
      margin-top: 12px;
    }}
    .profile-row small,
    .metric-label,
    .section-kicker,
    .control-block label,
    .controls label {{
      display: block;
      margin-bottom: 6px;
      color: var(--muted-soft);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 800;
    }}
    .profile-row strong {{
      display: block;
      font-size: 16px;
      line-height: 1.18;
      color: var(--text);
    }}
    .section {{
      margin-top: 16px;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 18px;
      margin-bottom: 12px;
    }}
    .section-head p {{
      max-width: 820px;
    }}
    .card {{
      padding: 18px 20px;
    }}
    .subtle {{
      color: var(--muted-soft);
      font-size: 13px;
    }}
    .toolbar,
    .controls {{
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248,250,252,0.98) 100%);
      box-shadow: var(--shadow);
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
      padding: 14px 15px;
      min-height: 94px;
    }}
    .metric-card strong {{
      display: block;
      font-size: 22px;
      line-height: 1.08;
      letter-spacing: -0.02em;
    }}
    .metric-card .sub,
    .metric-card .muted-note {{
      margin-top: 6px;
      font-size: 12px;
      line-height: 1.45;
      color: var(--muted);
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line-strong);
      background: rgba(255,255,255,0.86);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: var(--muted);
    }}
    .badge.warn {{
      border-color: rgba(180, 35, 24, 0.22);
      color: var(--danger);
      background: rgba(180, 35, 24, 0.05);
    }}
    .summary-table,
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 12px;
    }}
    th, td {{
      border-top: 1px solid #e8edf4;
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
      word-break: break-word;
    }}
    th {{
      background: #f7f9fc;
      color: var(--muted-soft);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 11px;
      font-weight: 800;
    }}
    select, input, button {{
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line-strong);
      border-radius: 11px;
      background: #fff;
      color: var(--text);
      font-size: 14px;
      box-sizing: border-box;
    }}
    button {{
      cursor: pointer;
      font-weight: 700;
      background: #f8fafc;
    }}
    .pill-button {{
      border-radius: 999px;
    }}
    .chart-frame,
    .aggregate-chart,
    #basketPathChart,
    #basketContractTrend,
    #contractTrend {{
      width: 100%;
      border: 1px solid #dfe6ef;
      border-radius: 20px;
      background: linear-gradient(180deg, #fcfdff 0%, #f4f7fb 100%);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.75);
    }}
    .muted-list, .muted-note, .chain-note, .meta {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .file-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .file-link {{
      display: block;
      padding: 12px 14px;
      border-radius: var(--radius-md);
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
      text-decoration: none;
      color: var(--text);
      font-weight: 700;
    }}
    .file-link span {{
      display: block;
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
    }}
    .doc-layout {{
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      gap: 20px;
      align-items: start;
    }}
    .toc-panel {{
      position: sticky;
      top: 96px;
      padding: 16px 18px;
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .toc-panel ul {{
      margin: 0;
      padding-left: 18px;
    }}
    .toc-panel li {{
      margin: 6px 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .doc-article {{
      padding: 22px 24px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .doc-article h2 {{
      margin-top: 30px;
      padding-top: 16px;
      border-top: 1px solid #edf1f6;
    }}
    .doc-article details {{
      margin: 12px 0;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #fafcff;
    }}
    .doc-article summary {{
      cursor: pointer;
      font-weight: 700;
      color: var(--text);
    }}
    .footer-note {{
      margin-top: 16px;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #f7fafc;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 1220px) {{
      .shell {{
        width: min(1480px, 96vw);
      }}
      .hero-grid,
      .doc-layout {{
        grid-template-columns: 1fr;
      }}
      .toc-panel {{
        position: static;
      }}
      .metric-grid,
      .file-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 760px) {{
      .shell {{
        width: 95vw;
      }}
      .nav-shell {{
        position: static;
        flex-direction: column;
        align-items: flex-start;
      }}
      .hero,
      .card,
      .panel,
      .doc-article {{
        padding: 16px;
      }}
      .metric-grid,
      .file-grid {{
        grid-template-columns: 1fr;
      }}
      h1 {{
        font-size: 32px;
      }}
    }}
    {extra}
    """.strip()


def _site_nav(active: str, note: str = "Static research terminal") -> str:
    items = [
        ("Dashboard", "index.html"),
        ("Explorer", "explorer.html"),
        ("Baskets", "baskets.html"),
        ("Methodology", "methodology.html"),
    ]
    links = []
    for label, href in items:
        active_class = "active" if label.lower() == active.lower() else ""
        links.append(f'<a class="{active_class}" href="{href}">{html.escape(label)}</a>')
    return f"""
  <div class="topline"></div>
  <div class="shell">
    <header class="nav-shell">
      <div class="brand-lockup">
        <div class="brand-mark">I</div>
        <div class="brand-copy">
          <strong>Itô Markets</strong>
          <span>{html.escape(note)}</span>
        </div>
      </div>
      <nav class="nav-links">
        {"".join(links)}
      </nav>
    </header>
"""


def _dashboard_chart_script() -> str:
    return """
(() => {
  const STORAGE_KEY = "prediction-basket-dashboard-state-v3";
  const BL_DATA = JSON.parse(document.getElementById("basket-level-data").textContent);
  const basketPathSelect = document.getElementById("basketPathSelect");
  const basketPrevBtn = document.getElementById("basketPrevBtn");
  const basketNextBtn = document.getElementById("basketNextBtn");
  const basketHeroCode = document.getElementById("basketHeroCode");
  const basketHeroName = document.getElementById("basketHeroName");
  const basketHeroLevel = document.getElementById("basketHeroLevel");
  const basketHeroChange = document.getElementById("basketHeroChange");
  const basketHeroAsOf = document.getElementById("basketHeroAsOf");
  const basketHeroNarrative = document.getElementById("basketHeroNarrative");
  const basketHeroBadges = document.getElementById("basketHeroBadges");
  const basketProfileGrid = document.getElementById("basketProfileGrid");
  const basketPathMeta = document.getElementById("basketPathMeta");
  const basketPathChart = document.getElementById("basketPathChart");
  const dateModeButtons = Array.from(document.querySelectorAll("[data-date-mode]"));
  const scaleModeButtons = Array.from(document.querySelectorAll("[data-scale-mode]"));
  const rangeButtons = Array.from(document.querySelectorAll("[data-range]"));
  const toggleCash = document.getElementById("toggleCash");
  const toggleTurnover = document.getElementById("toggleTurnover");
  const toggleRebalances = document.getElementById("toggleRebalances");
  const toggleEntriesExits = document.getElementById("toggleEntriesExits");

  const DEFAULT_STATE = {
    basket: null,
    dateMode: "since_inception",
    scaleMode: "normalized",
    range: "ALL",
    showCash: false,
    showTurnover: false,
    showRebalances: true,
    showEntriesExits: false
  };

  let chart = null;

  function esc(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function fmtPct(value) {
    return (Number(value || 0) * 100).toFixed(2) + "%";
  }

  function fmtSignedPct(value) {
    const n = Number(value || 0) * 100;
    const prefix = n > 0 ? "+" : "";
    return prefix + n.toFixed(2) + "%";
  }

  function fmtSignedNumber(value) {
    const n = Number(value || 0);
    const prefix = n > 0 ? "+" : "";
    return prefix + n.toFixed(2);
  }

  function fmtDate(value) {
    return String(value || "").slice(0, 10);
  }

  function parseDate(value) {
    return new Date(fmtDate(value) + "T00:00:00Z");
  }

  function lookupWindowReturn(rows, days) {
    if (!rows.length) return null;
    const latest = rows[rows.length - 1];
    const latestNav = Number(latest.tradable_nav || latest.basket_level || 0);
    if (!(latestNav > 0)) return null;
    const cutoff = new Date(parseDate(latest.rebalance_date).getTime() - days * 24 * 60 * 60 * 1000);
    let anchor = null;
    rows.forEach(row => {
      if (parseDate(row.rebalance_date) <= cutoff) {
        anchor = row;
      }
    });
    const anchorNav = Number(anchor?.tradable_nav || anchor?.basket_level || 0);
    if (!(anchorNav > 0)) return null;
    return latestNav / anchorNav - 1;
  }

  function lookupYtdReturn(rows) {
    if (!rows.length) return null;
    const latest = rows[rows.length - 1];
    const year = fmtDate(latest.rebalance_date).slice(0, 4);
    const anchor = rows.find(row => fmtDate(row.rebalance_date) >= `${year}-01-01`);
    const latestNav = Number(latest.tradable_nav || latest.basket_level || 0);
    const anchorNav = Number(anchor?.tradable_nav || anchor?.basket_level || 0);
    if (!(latestNav > 0) || !(anchorNav > 0)) return null;
    return latestNav / anchorNav - 1;
  }

  function basketCodes() {
    const ordered = Array.isArray(BL_DATA.order) ? BL_DATA.order.filter(code => BL_DATA.series?.[code]) : [];
    return ordered.length ? ordered : Object.keys(BL_DATA.series || {}).sort();
  }

  function loadState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return {};
      return JSON.parse(raw) || {};
    } catch (err) {
      return {};
    }
  }

  function saveState(state) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (err) {
      return;
    }
  }

  const state = Object.assign({}, DEFAULT_STATE, loadState());
  const codes = basketCodes();
  if (!codes.includes(state.basket)) {
    state.basket = codes[0] || null;
  }

  function currentSeries() {
    return state.basket ? BL_DATA.series?.[state.basket] || null : null;
  }

  function activateButtons(buttons, activeValue, attrName) {
    buttons.forEach(btn => {
      const selected = String(btn.getAttribute(attrName)) === String(activeValue);
      btn.classList.toggle("active", selected);
    });
  }

  function syncControls() {
    basketPathSelect.value = state.basket || "";
    activateButtons(dateModeButtons, state.dateMode, "data-date-mode");
    activateButtons(scaleModeButtons, state.scaleMode, "data-scale-mode");
    activateButtons(rangeButtons, state.range, "data-range");
    toggleCash.checked = !!state.showCash;
    toggleTurnover.checked = !!state.showTurnover;
    toggleRebalances.checked = !!state.showRebalances;
    toggleEntriesExits.checked = !!state.showEntriesExits;
  }

  function applyDateMode(rows, series) {
    if (!Array.isArray(rows) || !rows.length) return [];
    if (state.dateMode !== "since_inception" || !series?.effective_inception_date) return rows.slice();
    const filtered = rows.filter(row => fmtDate(row.rebalance_date) >= fmtDate(series.effective_inception_date));
    return filtered.length ? filtered : rows.slice();
  }

  function applyRange(rows) {
    if (!rows.length || state.range === "ALL") return rows.slice();
    const daysByRange = {
      "3M": 92,
      "6M": 183,
      "1Y": 366,
      "2Y": 731
    };
    const lookbackDays = daysByRange[state.range];
    if (!lookbackDays) return rows.slice();
    const lastDate = parseDate(rows[rows.length - 1].rebalance_date);
    const threshold = new Date(lastDate.getTime() - lookbackDays * 24 * 60 * 60 * 1000);
    const filtered = rows.filter(row => parseDate(row.rebalance_date) >= threshold);
    return filtered.length ? filtered : rows.slice();
  }

  function visibleRows(series) {
    const rawRows = (series?.rows || []).slice().sort((a, b) => String(a.rebalance_date).localeCompare(String(b.rebalance_date)));
    return applyRange(applyDateMode(rawRows, series));
  }

  function normalizeSeries(rows, field) {
    const values = rows.map(row => Number(row[field] || 0));
    if (state.scaleMode !== "normalized") {
      return values;
    }
    const baseRow = rows.find(row => Number.isFinite(Number(row[field])) && Number(row[field]) > 0);
    const base = baseRow ? Number(baseRow[field]) : 100;
    return values.map(value => (Number(value || 0) / base) * 100);
  }

  function normalizeNullableSeries(rows, rawValues, baseField) {
    if (state.scaleMode !== "normalized") {
      return rawValues.slice();
    }
    const baseRow = rows.find(row => Number.isFinite(Number(row[baseField])) && Number(row[baseField]) > 0);
    const base = baseRow ? Number(baseRow[baseField]) : 100;
    return rawValues.map(value => {
      const n = Number(value);
      if (!Number.isFinite(n) || n <= 0) return null;
      return (n / base) * 100;
    });
  }

  function warningBadges(latest) {
    const warnings = [];
    if (Number(latest.tail_probability_weight_share || 0) >= 0.40) warnings.push("Tail-heavy");
    if (Number(latest.price_coverage_share || 1) < 0.65) warnings.push("Low coverage");
    if (Number(latest.slot_coverage_ratio || 1) < 0.75) warnings.push("Missing slots");
    if (Number(latest.proxy_weight_share || 0) > 0.35) warnings.push("Proxy-dominated");
    return warnings;
  }

  function renderHero(series, allRows, rows) {
    if (!series || !rows.length) {
      basketHeroCode.textContent = "No basket";
      basketHeroName.textContent = "No basket path data available.";
      basketHeroLevel.textContent = "—";
      basketHeroChange.textContent = "";
      basketHeroAsOf.textContent = "";
      basketHeroNarrative.textContent = "";
      basketHeroBadges.innerHTML = "";
      basketProfileGrid.innerHTML = "";
      return;
    }
    const latest = rows[rows.length - 1];
    const previous = rows.length > 1 ? rows[rows.length - 2] : null;
    const latestNav = Number(latest.tradable_nav || latest.basket_level || 0);
    const prevNav = Number(previous?.tradable_nav || previous?.basket_level || latestNav || 0);
    const oneDay = previous && prevNav > 0 ? latestNav / prevNav - 1 : null;
    const thirtyDay = lookupWindowReturn(allRows, 30);
    const ytd = lookupYtdReturn(allRows);
    const warnings = warningBadges(latest);
    basketHeroCode.textContent = `${series.domain} | ${series.code}`;
    basketHeroName.textContent = series.basket_name;
    basketHeroLevel.textContent = latestNav ? latestNav.toFixed(2) : "—";
    basketHeroChange.innerHTML = `${fmtSignedNumber(latestNav - prevNav)} <span>${oneDay == null ? "—" : fmtSignedPct(oneDay)}</span>`;
    basketHeroChange.className = `hero-change ${(oneDay || 0) >= 0 ? "up" : "down"}`;
    basketHeroAsOf.textContent = `As of ${fmtDate(latest.rebalance_date)} · Default view starts ${series.effective_inception_date || series.raw_history_start || "—"}`;
    basketHeroNarrative.textContent = `Tradable NAV for ${series.basket_name} with daily history reconstructed from selected contracts, explicit cash handling, and visible gap overlays only where the real basket path is fully in cash and unpriced.`;
    basketHeroBadges.innerHTML = [
      `<span class="badge">${esc(oneDay == null ? "1D —" : `1D ${fmtSignedPct(oneDay)}`)}</span>`,
      `<span class="badge">${esc(thirtyDay == null ? "30D —" : `30D ${fmtSignedPct(thirtyDay)}`)}</span>`,
      `<span class="badge">${esc(ytd == null ? "YTD —" : `YTD ${fmtSignedPct(ytd)}`)}</span>`,
      ...warnings.map(text => `<span class="badge warn">${esc(text)}</span>`)
    ].join("");
    basketProfileGrid.innerHTML = `
      <div class="profile-row"><small>History Window</small><strong>${esc(series.raw_history_start || "—")} → ${esc(fmtDate(latest.rebalance_date))}</strong></div>
      <div class="profile-row"><small>Effective Inception</small><strong>${esc(series.effective_inception_date || "—")}</strong></div>
      <div class="profile-row"><small>Coverage At Inception</small><strong>${series.price_coverage_at_inception == null ? "—" : esc(fmtPct(series.price_coverage_at_inception))}</strong></div>
      <div class="profile-row"><small>Current Coverage</small><strong>${latest.price_coverage_share == null ? "—" : esc(fmtPct(latest.price_coverage_share))}</strong></div>
      <div class="profile-row"><small>Current Cash</small><strong>${esc(fmtPct(latest.cash_weight || 0))}</strong></div>
      <div class="profile-row"><small>Tail / Slot Coverage</small><strong>${latest.tail_probability_weight_share == null ? "—" : esc(fmtPct(latest.tail_probability_weight_share))} / ${latest.slot_coverage_ratio == null ? "—" : esc(fmtPct(latest.slot_coverage_ratio))}</strong></div>
    `;
  }

  function renderMeta(series, rows) {
    if (!series || !rows.length) {
      basketPathMeta.innerHTML = "<div class='metric-card'><span class='metric-label'>Status</span><strong>No basket path series available.</strong></div>";
      return;
    }
    const latest = rows[rows.length - 1];
    const avgCash = rows.reduce((acc, row) => acc + Number(row.cash_weight || 0), 0) / Math.max(rows.length, 1);
    const latestCash = Number(rows[rows.length - 1].cash_weight || 0);
    const totalTurnover = rows.reduce((acc, row) => acc + Number(row.turnover || 0), 0);
    const cards = [
      ["Current NAV", Number(latest.tradable_nav || latest.basket_level || 0).toFixed(2)],
      ["Avg Cash", fmtPct(avgCash)],
      ["Latest Cash", fmtPct(latest.cash_weight)],
      ["Current Coverage", latest.price_coverage_share == null ? "—" : fmtPct(latest.price_coverage_share)],
      ["Tail Share", latest.tail_probability_weight_share == null ? "—" : fmtPct(latest.tail_probability_weight_share)],
      ["Slot Coverage", latest.slot_coverage_ratio == null ? "—" : fmtPct(latest.slot_coverage_ratio)],
      ["Proxy Share", latest.proxy_weight_share == null ? "—" : fmtPct(latest.proxy_weight_share)],
      ["Total Turnover", totalTurnover.toFixed(3)]
    ];
    const badges = warningBadges(latest);
    const badgeHtml = badges.length
      ? `<div class="metric-card"><span class="metric-label">Warnings</span><strong>${badges.map(esc).join(" · ")}</strong><div class="sub">Tail ${fmtPct(latest.tail_probability_weight_share || 0)} · Proxy ${fmtPct(latest.proxy_weight_share || 0)}</div></div>`
      : `<div class="metric-card"><span class="metric-label">Status</span><strong>Tradable NAV active</strong><div class="sub">Tail ${fmtPct(latest.tail_probability_weight_share || 0)} · Proxy ${fmtPct(latest.proxy_weight_share || 0)}</div></div>`;
    basketPathMeta.innerHTML = cards.map(([label, value]) =>
      "<div class='metric-card'><span class='metric-label'>" + esc(label) + "</span><strong>" + esc(value) + "</strong></div>"
    ).join("") + badgeHtml;
  }

  function buildOption(series, rows) {
    const navValues = normalizeSeries(rows, "tradable_nav");
    const overlayRaw = rows.map(row => row.graph_overlay_nav == null ? null : Number(row.graph_overlay_nav));
    const overlayValues = normalizeNullableSeries(rows, overlayRaw, "tradable_nav");
    const changes = rows.map(row => Number(row.entries || 0) + Number(row.exits || 0));
    const navDisplayMap = new Map(rows.map((row, idx) => [fmtDate(row.rebalance_date), navValues[idx]]));
    const overlayDisplayMap = new Map(rows.map((row, idx) => [fmtDate(row.rebalance_date), overlayValues[idx]]));
    const rowMap = new Map(rows.map(row => [fmtDate(row.rebalance_date), row]));
    const firstVisible = rows[0] ? fmtDate(rows[0].rebalance_date) : "";
    const canShadePrehistory = state.dateMode === "full_history" && series.effective_inception_date && firstVisible < fmtDate(series.effective_inception_date);

    const seriesList = [
      {
        name: state.scaleMode === "normalized" ? "Tradable NAV (Normalized)" : "Tradable NAV",
        type: "line",
        smooth: false,
        showSymbol: false,
        symbolSize: 6,
        sampling: "lttb",
        lineStyle: { width: 2.8, color: "#1d4ed8" },
        areaStyle: { color: "rgba(59, 130, 246, 0.10)" },
        emphasis: { focus: "series" },
        data: rows.map((row, idx) => [fmtDate(row.rebalance_date), navValues[idx]]),
        markLine: {
          symbol: ["none", "none"],
          animation: false,
          label: { show: false },
          lineStyle: { color: "rgba(148, 163, 184, 0.55)", width: 1, type: "dashed" },
          data: []
        },
        markArea: canShadePrehistory ? {
          silent: true,
          itemStyle: { color: "rgba(148, 163, 184, 0.10)" },
          label: { show: true, color: "#475569", fontSize: 11 },
          data: [[
            { name: "Inactive prehistory", xAxis: firstVisible },
            { xAxis: fmtDate(series.effective_inception_date) }
          ]]
        } : undefined
      }
    ];

    if (overlayValues.some(value => value != null)) {
      seriesList.push({
        name: state.scaleMode === "normalized" ? "Gap Overlay (Normalized)" : "Gap Overlay",
        type: "line",
        smooth: true,
        showSymbol: false,
        silent: true,
        connectNulls: false,
        lineStyle: { width: 1.8, color: "rgba(15, 23, 42, 0.42)", type: "dashed" },
        data: rows.map((row, idx) => [fmtDate(row.rebalance_date), overlayValues[idx]])
      });
    }

    if (state.showRebalances) {
      const rebalanceLines = rows
        .filter(row => row.rebalanced || Number(row.entries || 0) + Number(row.exits || 0) > 0)
        .map(row => ({ xAxis: fmtDate(row.rebalance_date) }));
      if (canShadePrehistory) {
        rebalanceLines.push({
          xAxis: fmtDate(series.effective_inception_date),
          lineStyle: { color: "#0f172a", width: 1.4, type: "solid" },
          label: { show: true, formatter: "Inception", color: "#0f172a" }
        });
      }
      seriesList[0].markLine.data = rebalanceLines;
    }

    if (state.showCash) {
      seriesList.push({
        name: "Cash Weight",
        type: "line",
        smooth: true,
        yAxisIndex: 1,
        showSymbol: false,
        lineStyle: { width: 2.0, color: "#0f9f6e" },
        data: rows.map(row => [fmtDate(row.rebalance_date), Number(row.cash_weight || 0) * 100])
      });
    }

    if (state.showTurnover) {
      seriesList.push({
        name: "Turnover",
        type: "line",
        smooth: true,
        yAxisIndex: 2,
        showSymbol: false,
        lineStyle: { width: 1.9, color: "#7c3aed" },
        areaStyle: { color: "rgba(124, 58, 237, 0.06)" },
        data: rows.map(row => [fmtDate(row.rebalance_date), Number(row.turnover || 0)])
      });
    }

    if (state.showEntriesExits) {
      seriesList.push({
        name: "Entries + Exits",
        type: "bar",
        yAxisIndex: 2,
        barMaxWidth: 12,
        itemStyle: { color: "rgba(148, 163, 184, 0.55)" },
        data: rows.map((row, idx) => [fmtDate(row.rebalance_date), changes[idx]])
      });
    }

    return {
      animation: false,
      color: ["#1d4ed8", "rgba(15, 23, 42, 0.42)", "#0f9f6e", "#7c3aed", "#94a3b8"],
      grid: { left: 58, right: 92, top: 34, bottom: 72 },
      legend: {
        top: 0,
        left: 0,
        textStyle: { fontFamily: "IBM Plex Sans, Segoe UI, Arial, sans-serif" }
      },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "cross" },
        backgroundColor: "rgba(15, 23, 42, 0.95)",
        borderWidth: 0,
        textStyle: { color: "#f8fafc" },
        formatter(params) {
          const list = Array.isArray(params) ? params : [params];
          const first = list[0] || {};
          const date = fmtDate(Array.isArray(first.value) ? first.value[0] : first.axisValue);
          const row = rowMap.get(date);
          const inceptionStatus = series.effective_inception_date && date < fmtDate(series.effective_inception_date)
            ? "Inactive prehistory"
            : "Active history";
          const lines = [
            "<strong>" + esc(date) + "</strong>",
            "Status: " + esc(inceptionStatus),
            "Tradable NAV: " + Number(navDisplayMap.get(date) || 0).toFixed(2)
          ];
          if (row) {
            lines.push("Cash: " + fmtPct(row.cash_weight));
            lines.push("Turnover: " + Number(row.turnover || 0).toFixed(3));
            lines.push("Entries: " + Number(row.entries || 0));
            lines.push("Exits: " + Number(row.exits || 0));
            lines.push("Coverage: " + (row.price_coverage_share == null ? "—" : fmtPct(row.price_coverage_share)));
            if (overlayDisplayMap.get(date) != null) {
              lines.push("Gap Overlay: " + Number(overlayDisplayMap.get(date) || 0).toFixed(2) + " (graph only)");
            }
            lines.push("Tail Share: " + (row.tail_probability_weight_share == null ? "—" : fmtPct(row.tail_probability_weight_share)));
            lines.push("Slot Coverage: " + (row.slot_coverage_ratio == null ? "—" : fmtPct(row.slot_coverage_ratio)));
          }
          return lines.join("<br/>");
        }
      },
      xAxis: {
        type: "time",
        axisLabel: { color: "#64748b" },
        axisLine: { lineStyle: { color: "#cbd5e1" } },
        splitLine: { show: false }
      },
      yAxis: [
        {
          type: "value",
          name: state.scaleMode === "normalized" ? "Normalized Index" : "Level",
          nameTextStyle: { color: "#475569" },
          scale: true,
          axisLabel: { color: "#64748b" },
          splitLine: { lineStyle: { color: "rgba(203, 213, 225, 0.55)" } }
        },
        {
          type: "value",
          name: "Cash %",
          nameTextStyle: { color: "#475569" },
          min: 0,
          max: 100,
          position: "right",
          show: state.showCash,
          axisLabel: { formatter: "{value}%", color: "#64748b" },
          splitLine: { show: false }
        },
        {
          type: "value",
          name: state.showTurnover ? "Turnover" : "Entries / Exits",
          nameTextStyle: { color: "#475569" },
          min: 0,
          position: "right",
          offset: state.showCash ? 64 : 0,
          show: state.showTurnover || state.showEntriesExits,
          axisLabel: { color: "#64748b" },
          splitLine: { show: false }
        }
      ],
      dataZoom: [
        { type: "inside", throttle: 50 },
        { type: "slider", height: 18, bottom: 18, borderColor: "#d6dee9", fillerColor: "rgba(29, 78, 216, 0.10)" }
      ],
      series: seriesList
    };
  }

  function render() {
    const series = currentSeries();
      if (!series || !Array.isArray(series.rows) || !series.rows.length) {
      basketPathMeta.innerHTML = "<div class='metric-card'><span class='metric-label'>Status</span><strong>No basket path data available.</strong></div>";
      if (chart) chart.clear();
      return;
    }
    const allRows = (series?.rows || []).slice().sort((a, b) => String(a.rebalance_date).localeCompare(String(b.rebalance_date)));
    const rows = visibleRows(series);
    renderHero(series, allRows, rows);
    renderMeta(series, rows);
    if (!chart) {
      chart = echarts.init(basketPathChart, null, { renderer: "canvas" });
      window.addEventListener("resize", () => chart && chart.resize());
    }
    chart.setOption(buildOption(series, rows), true);
    syncControls();
    saveState(state);
  }

  basketPathSelect.innerHTML = codes.map(code => {
    const series = BL_DATA.series?.[code];
    const label = series ? `${series.domain} | ${series.code} | ${series.basket_name}` : code;
    return `<option value="${esc(code)}">${esc(label)}</option>`;
  }).join("");

  basketPathSelect.addEventListener("change", () => {
    state.basket = basketPathSelect.value;
    render();
  });

  basketPrevBtn.addEventListener("click", () => {
    const idx = codes.indexOf(state.basket);
    if (idx > 0) {
      state.basket = codes[idx - 1];
      render();
    }
  });

  basketNextBtn.addEventListener("click", () => {
    const idx = codes.indexOf(state.basket);
    if (idx >= 0 && idx < codes.length - 1) {
      state.basket = codes[idx + 1];
      render();
    }
  });

  dateModeButtons.forEach(btn => btn.addEventListener("click", () => {
    state.dateMode = btn.getAttribute("data-date-mode");
    render();
  }));

  scaleModeButtons.forEach(btn => btn.addEventListener("click", () => {
    state.scaleMode = btn.getAttribute("data-scale-mode");
    render();
  }));

  rangeButtons.forEach(btn => btn.addEventListener("click", () => {
    state.range = btn.getAttribute("data-range");
    render();
  }));

  toggleCash.addEventListener("change", () => {
    state.showCash = toggleCash.checked;
    render();
  });
  toggleTurnover.addEventListener("change", () => {
    state.showTurnover = toggleTurnover.checked;
    render();
  });
  toggleRebalances.addEventListener("change", () => {
    state.showRebalances = toggleRebalances.checked;
    render();
  });
  toggleEntriesExits.addEventListener("change", () => {
    state.showEntriesExits = toggleEntriesExits.checked;
    render();
  });

  syncControls();
  render();
})();
""".strip()


def _explorer_chart_script() -> str:
    return """
(() => {
  const DATA = JSON.parse(document.getElementById("explorer-data").textContent);
  const state = {
    basket: DATA.baskets.length ? DATA.baskets[0].code : null,
    month: DATA.months.length ? DATA.months[DATA.months.length - 1] : null,
    contract: null,
    query: ""
  };

  const basketSelect = document.getElementById("basketSelect");
  const monthSelect = document.getElementById("monthSelect");
  const searchInput = document.getElementById("searchInput");
  const prevMonthBtn = document.getElementById("prevMonthBtn");
  const nextMonthBtn = document.getElementById("nextMonthBtn");
  const clearSearchBtn = document.getElementById("clearSearchBtn");
  const contractsBody = document.getElementById("contractsBody");
  const leftMeta = document.getElementById("leftMeta");
  const detailTitle = document.getElementById("detailTitle");
  const detailMeta = document.getElementById("detailMeta");
  const detailStats = document.getElementById("detailStats");
  const contractTrend = document.getElementById("contractTrend");
  const weightsGrid = document.getElementById("weightsGrid");
  const chainGrid = document.getElementById("chainGrid");
  const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const trendChart = echarts.init(contractTrend, null, { renderer: "canvas" });
  window.addEventListener("resize", () => trendChart.resize());

  function esc(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function fmtPct(value) {
    return (Number(value || 0) * 100).toFixed(2) + "%";
  }

  function fmtCompactPct(value) {
    return (Number(value || 0) * 100).toFixed(1) + "%";
  }

  function fmtMonth(month) {
    if (!month) return "—";
    const [year, mm] = String(month).split("-");
    const idx = Math.max(0, Math.min(11, Number(mm || 1) - 1));
    return `${MONTH_NAMES[idx]} ${year}`;
  }

  function currentBasketBlock() {
    return DATA.contracts_by_basket[state.basket] || null;
  }

  function currentMonthRows() {
    const block = currentBasketBlock();
    if (!block) return [];
    const rows = (block.months[state.month] || []).slice();
    if (!state.query.trim()) return rows;
    const q = state.query.trim().toLowerCase();
    return rows.filter(row => row.market_id.toLowerCase().includes(q) || row.title.toLowerCase().includes(q));
  }

  function monthIndex(month) {
    return DATA.months.indexOf(month);
  }

  function ensureContractSelection(rows) {
    if (!rows.length) {
      state.contract = null;
      return;
    }
    if (!state.contract || !rows.some(row => row.market_id === state.contract)) {
      state.contract = rows[0].market_id;
    }
  }

  function renderContractRows() {
    const rows = currentMonthRows();
    ensureContractSelection(rows);
    const block = currentBasketBlock();
    const meta = block ? block.meta : null;
    leftMeta.textContent = meta
      ? `${meta.domain} | ${meta.code} | Month ${state.month} | Contracts: ${rows.length}`
      : "No basket selected.";

    const maxW = Math.max(...rows.map(row => Number(row.weight || 0)), 0.000001);
    if (!rows.length) {
      contractsBody.innerHTML = `<tr><td colspan="4" class="muted-note">No contracts match this basket, month, and search filter.</td></tr>`;
      return;
    }
    contractsBody.innerHTML = rows.map((row, idx) => {
      const activeClass = row.market_id === state.contract ? "active" : "";
      const barW = Math.max(2, (Number(row.weight || 0) / maxW) * 100);
      return `<tr class="contract-row ${activeClass}" data-market-id="${esc(row.market_id)}">
        <td>${idx + 1}</td>
        <td><div class="contract-title">${esc(row.title)}</div><code class="contract-id">${esc(row.market_id)}</code><div class="chain-note">${esc(row.position_instruction || "LONG_YES")} · Market YES ${row.market_yes_price == null ? "—" : fmtCompactPct(row.market_yes_price)} · Risk ${row.effective_risk_price == null ? "—" : fmtCompactPct(row.effective_risk_price)}</div></td>
        <td class="weight-cell">${fmtPct(row.weight)}<div class="bar"><span style="width:${barW}%;"></span></div></td>
        <td>${esc(row.end_date || "—")}</td>
      </tr>`;
    }).join("");
  }

  function renderStats(weights, contract) {
    const active = DATA.months.filter(month => Number(contract.weights[month] || 0) > 0);
    const currentWeight = Number(contract.weights[state.month] || 0);
    const maxWeight = Math.max(...weights, 0);
    const peakIdx = weights.indexOf(maxWeight);
    const peakMonth = peakIdx >= 0 ? DATA.months[peakIdx] : null;
    const firstActive = active.length ? active[0] : null;
    const lastActive = active.length ? active[active.length - 1] : null;
    detailStats.innerHTML = `
      <div class="stat-card">
        <span class="label">Selected Month</span>
        <div class="value">${fmtCompactPct(currentWeight)}</div>
        <div class="sub">${fmtMonth(state.month)}</div>
      </div>
      <div class="stat-card">
        <span class="label">Average Weight</span>
        <div class="value">${fmtCompactPct(contract.avg_weight)}</div>
        <div class="sub">Across active months</div>
      </div>
      <div class="stat-card">
        <span class="label">Peak Weight</span>
        <div class="value">${fmtCompactPct(maxWeight)}</div>
        <div class="sub">${peakMonth ? fmtMonth(peakMonth) : "—"}</div>
      </div>
      <div class="stat-card">
        <span class="label">Active Months</span>
        <div class="value">${active.length}</div>
        <div class="sub">${firstActive ? fmtMonth(firstActive) : "—"} to ${lastActive ? fmtMonth(lastActive) : "—"}</div>
      </div>
      <div class="stat-card">
        <span class="label">Ticker Chain</span>
        <div class="value">${Number(contract.chain_market_count || 0)}</div>
        <div class="sub">${esc(contract.ticker_name || "No ticker label")}</div>
      </div>
      <div class="stat-card">
        <span class="label">Direction</span>
        <div class="value">${esc(contract.position_instruction || "LONG_YES")}</div>
        <div class="sub">${esc(contract.direction_reason || "risk-up mapping")}</div>
      </div>`;
  }

  function renderTrend(weights) {
    const block = currentBasketBlock();
    const contract = block && state.contract ? block.contracts[state.contract] : null;
    const firstActive = contract && contract.first_active_month ? contract.first_active_month : DATA.months[0];
    const startIdx = Math.max(0, DATA.months.indexOf(firstActive));
    const visibleMonths = DATA.months.slice(startIdx);
    const rows = visibleMonths.map((month, idx) => [month, Number(weights[startIdx + idx] || 0) * 100]);
    const selectedMonth = state.month;
    trendChart.setOption({
      animation: false,
      color: ["#1d4ed8"],
      grid: { left: 54, right: 18, top: 26, bottom: 54 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "line" },
        backgroundColor: "rgba(15, 23, 42, 0.95)",
        borderWidth: 0,
        textStyle: { color: "#f8fafc" },
        formatter(params) {
          const first = Array.isArray(params) ? params[0] : params;
          const value = Array.isArray(first.value) ? first.value[1] : 0;
          const date = Array.isArray(first.value) ? first.value[0] : "";
          return `<strong>${esc(fmtMonth(date))}</strong><br/>Weight: ${Number(value || 0).toFixed(2)}%`;
        }
      },
      xAxis: {
        type: "time",
        axisLabel: { color: "#64748b" },
        axisLine: { lineStyle: { color: "#cbd5e1" } },
        splitLine: { show: false }
      },
      yAxis: {
        type: "value",
        name: "Weight %",
        min: 0,
        axisLabel: { formatter: "{value}%", color: "#64748b" },
        splitLine: { lineStyle: { color: "rgba(203, 213, 225, 0.55)" } }
      },
      series: [
        {
          name: "Weight",
          type: "line",
          smooth: false,
          showSymbol: false,
          lineStyle: { width: 2.4, color: "#1d4ed8" },
          areaStyle: { color: "rgba(59, 130, 246, 0.10)" },
          data: rows,
          markLine: {
            symbol: ["none", "none"],
            lineStyle: { color: "#0f172a", type: "dashed", width: 1.2 },
            label: { formatter: "Selected", color: "#0f172a" },
            data: selectedMonth ? [{ xAxis: selectedMonth }] : []
          }
        }
      ]
    }, true);
  }

  function renderMonthlyMatrix(contract) {
    const years = [...new Set(DATA.months.map(month => String(month).slice(0, 4)))];
    const header = MONTH_NAMES.map(name => `<th>${name}</th>`).join("");
    const rows = years.map(year => {
      const cells = MONTH_NAMES.map((_, idx) => {
        const month = `${year}-${String(idx + 1).padStart(2, "0")}-01`;
        const val = Number(contract.weights[month] || 0);
        const activeClass = val > 0 ? "active-month" : "empty-month";
        const selectedClass = month === state.month ? " selected-month" : "";
        const text = val > 0 ? fmtCompactPct(val) : "—";
        return `<td class="${activeClass + selectedClass}">${text}</td>`;
      }).join("");
      return `<tr><td>${year}</td>${cells}</tr>`;
    }).join("");
    weightsGrid.innerHTML = `<table><thead><tr><th>Year</th>${header}</tr></thead><tbody>${rows}</tbody></table>`;
  }

  function renderChain(contract) {
    const chainRows = (contract.chain_markets || []).slice(0, 250);
    if (!chainRows.length) {
      chainGrid.innerHTML = `<p class="muted-note">No chain history loaded for this ticker.</p>`;
      return;
    }
    const body = chainRows.map((row, idx) => {
      const selectedBadge = row.selected_any ? `<span class="selected-pill">Selected</span>` : "";
      const selectedMonths = row.selected_rebalance_dates
        ? row.selected_rebalance_dates.split(",").map(value => value.trim()).filter(Boolean).map(fmtMonth).join(", ")
        : "—";
      return `<tr>
        <td>${idx + 1}</td>
        <td class="chain-market"><div class="contract-title">${esc(row.title || row.market_id)}</div><code>${esc(row.market_id)}</code><div class="chain-note">${selectedBadge}</div></td>
        <td>${esc(row.end_date || "—")}</td>
        <td>${esc(selectedMonths)}</td>
      </tr>`;
    }).join("");
    chainGrid.innerHTML = `<table>
      <thead><tr><th>#</th><th>Chain Market</th><th>End Date</th><th>Selected Months</th></tr></thead>
      <tbody>${body}</tbody>
    </table>`;
  }

  function renderContractDetail() {
    const block = currentBasketBlock();
    if (!block || !state.contract || !block.contracts[state.contract]) {
      detailTitle.textContent = "Contract Detail";
      detailMeta.textContent = "Select a contract on the left.";
      detailStats.innerHTML = "";
      trendChart.clear();
      weightsGrid.innerHTML = "";
      chainGrid.innerHTML = "";
      return;
    }

    const contract = block.contracts[state.contract];
    const weights = DATA.months.map(month => Number(contract.weights[month] || 0));
    const activeMonths = weights.filter(value => value > 0).length;
    detailTitle.innerHTML = esc(contract.title);
    detailMeta.innerHTML = `<code>${esc(contract.market_id)}</code><br/>Ticker: <code>${esc(contract.ticker_id)}</code> (${esc(contract.ticker_name || "n/a")}) | Chain Size: <strong>${Number(contract.chain_market_count || 0)}</strong> | Chain Window: <code>${esc(contract.chain_first_end_date || "—")}</code> → <code>${esc(contract.chain_last_end_date || "—")}</code> | Expiry: <code>${esc(contract.end_date || "—")}</code> | Instruction: <strong>${esc(contract.position_instruction || "LONG_YES")}</strong> | Market YES: <strong>${contract.market_yes_price == null ? "—" : fmtPct(contract.market_yes_price)}</strong> | Risk Price: <strong>${contract.effective_risk_price == null ? "—" : fmtPct(contract.effective_risk_price)}</strong> | Slot: <strong>${esc(contract.slot_name || "—")}</strong> | DTE band: <strong>${esc(contract.tenor_band_status || "—")}</strong> | Avg Weight: <strong>${fmtPct(contract.avg_weight)}</strong> | Active Months: <strong>${activeMonths}</strong>`;
    renderStats(weights, contract);
    renderTrend(weights);
    renderMonthlyMatrix(contract);
    renderChain(contract);
  }

  function renderAll() {
    renderContractRows();
    renderContractDetail();
    basketSelect.value = state.basket || "";
    monthSelect.value = state.month || "";
    prevMonthBtn.disabled = monthIndex(state.month) <= 0;
    nextMonthBtn.disabled = monthIndex(state.month) >= DATA.months.length - 1;
  }

  basketSelect.innerHTML = DATA.baskets.map(basket =>
    `<option value="${esc(basket.code)}">${esc(`${basket.domain} | ${basket.code} | ${basket.name}`)}</option>`
  ).join("");
  monthSelect.innerHTML = DATA.months.map(month => `<option value="${esc(month)}">${esc(month)}</option>`).join("");

  basketSelect.addEventListener("change", () => {
    state.basket = basketSelect.value;
    state.contract = null;
    renderAll();
  });
  monthSelect.addEventListener("change", () => {
    state.month = monthSelect.value;
    state.contract = null;
    renderAll();
  });
  searchInput.addEventListener("input", () => {
    state.query = searchInput.value || "";
    state.contract = null;
    renderAll();
  });
  clearSearchBtn.addEventListener("click", () => {
    state.query = "";
    searchInput.value = "";
    state.contract = null;
    renderAll();
  });
  prevMonthBtn.addEventListener("click", () => {
    const idx = monthIndex(state.month);
    if (idx > 0) {
      state.month = DATA.months[idx - 1];
      state.contract = null;
      renderAll();
    }
  });
  nextMonthBtn.addEventListener("click", () => {
    const idx = monthIndex(state.month);
    if (idx < DATA.months.length - 1) {
      state.month = DATA.months[idx + 1];
      state.contract = null;
      renderAll();
    }
  });
  contractsBody.addEventListener("click", ev => {
    const row = ev.target.closest("tr.contract-row");
    if (!row) return;
    state.contract = row.getAttribute("data-market-id");
    renderAll();
  });

  renderAll();
})();
""".strip()


def _baskets_page_script() -> str:
    return """
(() => {
  const DATA = JSON.parse(document.getElementById("basket-composition-data").textContent);
  const state = {
    basket: DATA.baskets.length ? DATA.baskets[0].code : null,
    contract: null,
    query: "",
    sort: "avg_weight"
  };

  const basketSelect = document.getElementById("basketsBasketSelect");
  const searchInput = document.getElementById("basketsSearchInput");
  const sortSelect = document.getElementById("basketsSortSelect");
  const contractsBody = document.getElementById("basketContractsBody");
  const basketMeta = document.getElementById("basketMeta");
  const basketChartImg = document.getElementById("basketCompositionChart");
  const basketMonthTable = document.getElementById("basketMonthTable");
  const detailTitle = document.getElementById("basketDetailTitle");
  const detailMeta = document.getElementById("basketDetailMeta");
  const detailStats = document.getElementById("basketDetailStats");
  const contractTrend = document.getElementById("basketContractTrend");
  const weightsGrid = document.getElementById("basketWeightsGrid");
  const chainGrid = document.getElementById("basketChainGrid");
  const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const trendChart = echarts.init(contractTrend, null, { renderer: "canvas" });
  window.addEventListener("resize", () => trendChart.resize());

  function esc(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function fmtPct(value) {
    return (Number(value || 0) * 100).toFixed(2) + "%";
  }

  function fmtCompactPct(value) {
    return (Number(value || 0) * 100).toFixed(1) + "%";
  }

  function fmtMonth(month) {
    if (!month) return "—";
    const [year, mm] = String(month).split("-");
    const idx = Math.max(0, Math.min(11, Number(mm || 1) - 1));
    return `${MONTH_NAMES[idx]} ${year}`;
  }

  function currentBasket() {
    return DATA.details?.[state.basket] || null;
  }

  function currentContracts() {
    const basket = currentBasket();
    if (!basket) return [];
    let rows = Object.values(basket.contracts || {});
    if (state.query.trim()) {
      const q = state.query.trim().toLowerCase();
      rows = rows.filter(row =>
        row.market_id.toLowerCase().includes(q) ||
        row.title.toLowerCase().includes(q) ||
        String(row.ticker_name || "").toLowerCase().includes(q)
      );
    }
    const sortKey = state.sort || "avg_weight";
    rows.sort((a, b) => Number(b[sortKey] || 0) - Number(a[sortKey] || 0));
    return rows;
  }

  function ensureContractSelection(rows) {
    if (!rows.length) {
      state.contract = null;
      return;
    }
    if (!state.contract || !rows.some(row => row.market_id === state.contract)) {
      state.contract = rows[0].market_id;
    }
  }

  function renderBasketHeader() {
    const basket = currentBasket();
    if (!basket) {
      basketMeta.innerHTML = "<div class='metric-card'><span class='metric-label'>Status</span><strong>No basket selected.</strong></div>";
      basketMonthTable.innerHTML = "";
      basketChartImg.removeAttribute("src");
      return;
    }
    const meta = basket.meta || {};
    const cards = [
      ["Basket", `${meta.domain} | ${meta.code}`],
      ["Name", meta.name || meta.code],
      ["Effective Inception", meta.effective_inception_date || "—"],
      ["Cash Sleeve", fmtPct(meta.cash_weight)],
      ["Contracts", String(Object.keys(basket.contracts || {}).length)],
      ["Basket Weight", Number(meta.basket_weight || 0).toFixed(4)],
      ["Current NAV", meta.tradable_nav == null ? "—" : Number(meta.tradable_nav).toFixed(2)],
      ["Price Coverage", meta.price_coverage_share == null ? "—" : fmtPct(meta.price_coverage_share)],
      ["Slot Coverage", meta.slot_coverage_ratio == null ? "—" : fmtPct(meta.slot_coverage_ratio)],
      ["Tail / Proxy", `${fmtPct(meta.tail_probability_weight_share || 0)} / ${fmtPct(meta.proxy_weight_share || 0)}`]
    ];
    basketMeta.innerHTML = cards.map(([label, value]) =>
      `<div class="metric-card"><span class="metric-label">${esc(label)}</span><strong>${esc(value)}</strong></div>`
    ).join("");
    if (basket.composition_chart) {
      basketChartImg.src = basket.composition_chart;
      basketChartImg.alt = `${meta.code} composition chart`;
    }
    basketMonthTable.innerHTML = `
      <table>
        <thead><tr><th>Date</th><th>Contracts</th><th>Avg Weight</th><th>Max Weight</th></tr></thead>
        <tbody>${(basket.monthly_summary_rows || []).map(row => `
          <tr>
            <td>${esc(row.rebalance_date)}</td>
            <td>${Number(row.n_contracts || 0)}</td>
            <td>${Number(row.avg_contract_weight || 0).toFixed(4)}</td>
            <td>${Number(row.max_contract_weight || 0).toFixed(4)}</td>
          </tr>`).join("")}</tbody>
      </table>`;
  }

  function renderContractList() {
    const rows = currentContracts();
    ensureContractSelection(rows);
    if (!rows.length) {
      contractsBody.innerHTML = `<tr><td colspan="5" class="muted-note">No contracts match this basket and search filter.</td></tr>`;
      return;
    }
    const maxWeight = Math.max(...rows.map(row => Number(row.avg_weight || 0)), 0.000001);
    contractsBody.innerHTML = rows.map((row, idx) => {
      const activeClass = row.market_id === state.contract ? "active" : "";
      const barW = Math.max(3, (Number(row.avg_weight || 0) / maxWeight) * 100);
      return `<tr class="contract-row ${activeClass}" data-market-id="${esc(row.market_id)}">
        <td>${idx + 1}</td>
        <td><div class="contract-title">${esc(row.title)}</div><code class="contract-id">${esc(row.market_id)}</code><div class="chain-note">${esc(row.position_instruction || "LONG_YES")} · Market YES ${row.market_yes_price == null ? "—" : fmtCompactPct(row.market_yes_price)} · Risk ${row.effective_risk_price == null ? "—" : fmtCompactPct(row.effective_risk_price)}</div></td>
        <td class="weight-cell">${fmtPct(row.avg_weight)}<div class="bar"><span style="width:${barW}%;"></span></div></td>
        <td>${esc(fmtMonth(row.first_active_month))}</td>
        <td>${esc(row.end_date || "—")}</td>
      </tr>`;
    }).join("");
  }

  function renderStats(contract, visibleMonths, visibleWeights) {
    const maxWeight = Math.max(...visibleWeights, 0);
    const peakIdx = visibleWeights.indexOf(maxWeight);
    const peakMonth = peakIdx >= 0 ? visibleMonths[peakIdx] : null;
    detailStats.innerHTML = `
      <div class="stat-card">
        <span class="label">Average Weight</span>
        <div class="value">${fmtCompactPct(contract.avg_weight)}</div>
        <div class="sub">Selected months only</div>
      </div>
      <div class="stat-card">
        <span class="label">Peak Weight</span>
        <div class="value">${fmtCompactPct(contract.peak_weight || maxWeight)}</div>
        <div class="sub">${peakMonth ? fmtMonth(peakMonth) : "—"}</div>
      </div>
      <div class="stat-card">
        <span class="label">Starts</span>
        <div class="value">${esc(fmtMonth(contract.first_active_month))}</div>
        <div class="sub">Trimmed chart start</div>
      </div>
      <div class="stat-card">
        <span class="label">Ends</span>
        <div class="value">${esc(fmtMonth(contract.last_active_month))}</div>
        <div class="sub">Last active month</div>
      </div>
      <div class="stat-card">
        <span class="label">Ticker Chain</span>
        <div class="value">${Number(contract.chain_market_count || 0)}</div>
        <div class="sub">${esc(contract.ticker_name || "No ticker label")}</div>
      </div>
      <div class="stat-card">
        <span class="label">Direction</span>
        <div class="value">${esc(contract.position_instruction || "LONG_YES")}</div>
        <div class="sub">${esc(contract.direction_reason || "risk-up mapping")}</div>
      </div>`;
  }

  function renderTrend(contract) {
    const firstActive = contract.first_active_month || DATA.months[0];
    const startIdx = Math.max(0, DATA.months.indexOf(firstActive));
    const visibleMonths = DATA.months.slice(startIdx);
    const visibleWeights = visibleMonths.map(month => Number(contract.weights[month] || 0) * 100);
    renderStats(contract, visibleMonths, visibleWeights);
    trendChart.setOption({
      animation: false,
      color: ["#1d4ed8"],
      grid: { left: 54, right: 18, top: 26, bottom: 54 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "line" },
        backgroundColor: "rgba(15, 23, 42, 0.95)",
        borderWidth: 0,
        textStyle: { color: "#f8fafc" },
        formatter(params) {
          const first = Array.isArray(params) ? params[0] : params;
          const date = Array.isArray(first.value) ? first.value[0] : "";
          const value = Array.isArray(first.value) ? first.value[1] : 0;
          return `<strong>${esc(fmtMonth(date))}</strong><br/>Weight: ${Number(value || 0).toFixed(2)}%`;
        }
      },
      xAxis: {
        type: "time",
        axisLabel: { color: "#64748b" },
        axisLine: { lineStyle: { color: "#cbd5e1" } },
        splitLine: { show: false }
      },
      yAxis: {
        type: "value",
        name: "Weight %",
        min: 0,
        axisLabel: { formatter: "{value}%", color: "#64748b" },
        splitLine: { lineStyle: { color: "rgba(203, 213, 225, 0.55)" } }
      },
      series: [
        {
          name: "Weight",
          type: "line",
          smooth: false,
          showSymbol: false,
          lineStyle: { width: 2.4, color: "#1d4ed8" },
          areaStyle: { color: "rgba(59, 130, 246, 0.10)" },
          data: visibleMonths.map((month, idx) => [month, visibleWeights[idx]])
        }
      ]
    }, true);
    return visibleMonths;
  }

  function renderMatrix(contract) {
    const years = [...new Set(DATA.months.map(month => String(month).slice(0, 4)))];
    const header = MONTH_NAMES.map(name => `<th>${name}</th>`).join("");
    const rows = years.map(year => {
      const cells = MONTH_NAMES.map((_, idx) => {
        const month = `${year}-${String(idx + 1).padStart(2, "0")}-01`;
        const val = Number(contract.weights[month] || 0);
        const activeClass = val > 0 ? "active-month" : "empty-month";
        const text = val > 0 ? fmtCompactPct(val) : "—";
        return `<td class="${activeClass}">${text}</td>`;
      }).join("");
      return `<tr><td>${year}</td>${cells}</tr>`;
    }).join("");
    weightsGrid.innerHTML = `<table><thead><tr><th>Year</th>${header}</tr></thead><tbody>${rows}</tbody></table>`;
  }

  function renderChain(contract) {
    const chainRows = (contract.chain_markets || []).slice(0, 250);
    if (!chainRows.length) {
      chainGrid.innerHTML = `<p class="muted-note">No chain history loaded for this ticker.</p>`;
      return;
    }
    chainGrid.innerHTML = `<table>
      <thead><tr><th>#</th><th>Chain Market</th><th>End Date</th><th>Selected Months</th></tr></thead>
      <tbody>${chainRows.map((row, idx) => {
        const selectedMonths = row.selected_rebalance_dates
          ? row.selected_rebalance_dates.split(",").map(value => value.trim()).filter(Boolean).map(fmtMonth).join(", ")
          : "—";
        const selectedBadge = row.selected_any ? `<span class="selected-pill">Selected</span>` : "";
        return `<tr>
          <td>${idx + 1}</td>
          <td class="chain-market"><div class="contract-title">${esc(row.title || row.market_id)}</div><code>${esc(row.market_id)}</code><div class="chain-note">${selectedBadge}</div></td>
          <td>${esc(row.end_date || "—")}</td>
          <td>${esc(selectedMonths)}</td>
        </tr>`;
      }).join("")}</tbody>
    </table>`;
  }

  function renderDetail() {
    const basket = currentBasket();
    if (!basket || !state.contract || !basket.contracts[state.contract]) {
      detailTitle.textContent = "Contract Detail";
      detailMeta.textContent = "Select a contract on the left.";
      detailStats.innerHTML = "";
      trendChart.clear();
      weightsGrid.innerHTML = "";
      chainGrid.innerHTML = "";
      return;
    }
    const contract = basket.contracts[state.contract];
    detailTitle.textContent = contract.title;
    detailMeta.innerHTML = `<code>${esc(contract.market_id)}</code><br/>Ticker: <code>${esc(contract.ticker_id)}</code> (${esc(contract.ticker_name || "n/a")}) | Chain Size: <strong>${Number(contract.chain_market_count || 0)}</strong> | Chain Window: <code>${esc(contract.chain_first_end_date || "—")}</code> → <code>${esc(contract.chain_last_end_date || "—")}</code> | Expiry: <code>${esc(contract.end_date || "—")}</code> | Instruction: <strong>${esc(contract.position_instruction || "LONG_YES")}</strong> | Market YES: <strong>${contract.market_yes_price == null ? "—" : fmtPct(contract.market_yes_price)}</strong> | Risk Price: <strong>${contract.effective_risk_price == null ? "—" : fmtPct(contract.effective_risk_price)}</strong> | Slot: <strong>${esc(contract.slot_name || "—")}</strong> | DTE band: <strong>${esc(contract.tenor_band_status || "—")}</strong>`;
    renderTrend(contract);
    renderMatrix(contract);
    renderChain(contract);
  }

  function renderAll() {
    basketSelect.value = state.basket || "";
    searchInput.value = state.query || "";
    sortSelect.value = state.sort || "avg_weight";
    renderBasketHeader();
    renderContractList();
    renderDetail();
  }

  basketSelect.innerHTML = DATA.baskets.map(basket =>
    `<option value="${esc(basket.code)}">${esc(`${basket.domain} | ${basket.code} | ${basket.name}`)}</option>`
  ).join("");

  basketSelect.addEventListener("change", () => {
    state.basket = basketSelect.value;
    state.contract = null;
    renderAll();
  });
  searchInput.addEventListener("input", () => {
    state.query = searchInput.value || "";
    state.contract = null;
    renderAll();
  });
  sortSelect.addEventListener("change", () => {
    state.sort = sortSelect.value || "avg_weight";
    state.contract = null;
    renderAll();
  });
  contractsBody.addEventListener("click", ev => {
    const row = ev.target.closest("tr.contract-row");
    if (!row) return;
    state.contract = row.getAttribute("data-market-id");
    renderAll();
  });

  renderAll();
})();
""".strip()


def _build_methodology_html(
    output_path: Path,
    specs: list[ThematicBasketSpec],
    summary: pd.DataFrame,
    compositions: pd.DataFrame,
    start: str,
    end: str,
    monthly_summary: pd.DataFrame | None = None,
    cash_positions: pd.DataFrame | None = None,
    transitions: pd.DataFrame | None = None,
    factor_exposure: pd.DataFrame | None = None,
    ticker_registry: pd.DataFrame | None = None,
    ticker_history: pd.DataFrame | None = None,
    lifecycle_events: pd.DataFrame | None = None,
    cost_model: pd.DataFrame | None = None,
    aggregate_series: pd.DataFrame | None = None,
    aggregate_level_chart_rel: str | None = None,
    aggregate_cash_chart_rel: str | None = None,
    inception_policy: pd.DataFrame | None = None,
) -> None:
    rows = []
    for spec in specs:
        inc = ", ".join(f"<code>{html.escape(p.pattern)}</code> ({p.weight:.1f})" for p in spec.include_patterns)
        exc = ", ".join(f"<code>{html.escape(p)}</code>" for p in spec.exclude_patterns) if spec.exclude_patterns else "None"
        cats = ", ".join(f"<code>{html.escape(c)}</code>" for c in spec.allowed_categories) if spec.allowed_categories else "None"
        cap_rules = (
            f"template≤{spec.max_per_template}, community≤{spec.max_per_community}, "
            f"event≤{spec.max_per_event_family}, exclusive≤{spec.max_per_exclusive_group}"
        )
        rows.append(
            f"<tr><td>{html.escape(spec.domain)}</td><td>{html.escape(spec.basket_code)}</td>"
            f"<td>{html.escape(spec.basket_name)}</td><td>{spec.target_contracts}</td>"
            f"<td>{spec.min_contracts}-{spec.max_contracts}</td><td>{spec.cash_weight:.0%}</td>"
            f"<td>{cats}</td><td>{cap_rules}</td><td>{inc}</td><td>{exc}</td></tr>"
        )

    n_contracts = int(compositions[~compositions["is_cash"]]["market_id"].nunique()) if not compositions.empty else 0
    n_baskets = int(summary["basket_code"].nunique()) if not summary.empty else 0
    n_rebalances = int(compositions["rebalance_date"].nunique()) if not compositions.empty else 0

    # Build compact tables used in the methodology page.
    ms = monthly_summary.copy() if isinstance(monthly_summary, pd.DataFrame) else pd.DataFrame()
    cp = cash_positions.copy() if isinstance(cash_positions, pd.DataFrame) else pd.DataFrame()
    tr = transitions.copy() if isinstance(transitions, pd.DataFrame) else pd.DataFrame()
    fx = factor_exposure.copy() if isinstance(factor_exposure, pd.DataFrame) else pd.DataFrame()
    tk = ticker_registry.copy() if isinstance(ticker_registry, pd.DataFrame) else pd.DataFrame()
    th = ticker_history.copy() if isinstance(ticker_history, pd.DataFrame) else pd.DataFrame()
    lc = lifecycle_events.copy() if isinstance(lifecycle_events, pd.DataFrame) else pd.DataFrame()
    cm = cost_model.copy() if isinstance(cost_model, pd.DataFrame) else pd.DataFrame()
    ag = aggregate_series.copy() if isinstance(aggregate_series, pd.DataFrame) else pd.DataFrame()
    ip = inception_policy.copy() if isinstance(inception_policy, pd.DataFrame) else pd.DataFrame()

    if not ms.empty:
        ms = ms.sort_values(["rebalance_date", "basket_code"])
    if not cp.empty:
        cp = cp.sort_values(["rebalance_date", "basket_code"])
    if not tr.empty:
        tr = tr.sort_values(["rebalance_date", "basket_code"])
    if not fx.empty:
        fx = fx.sort_values(["rebalance_date", "basket_code"])
    if not tk.empty:
        tk = tk.sort_values(["domain", "basket_code", "ticker_id", "market_id"])
    if not th.empty:
        th = th.sort_values(["ticker_id", "chain_order", "market_id"])
    if not lc.empty:
        lc = lc.sort_values(["rebalance_date", "basket_code", "action"])
    if not cm.empty:
        cm = cm.sort_values(["rebalance_date", "basket_code"])
    if not ag.empty:
        ag = ag.sort_values(["rebalance_date"])
    if not ip.empty:
        ip = ip.sort_values("basket_code")

    cash_summary = pd.DataFrame()
    if not cp.empty:
        cash_summary = (
            cp.groupby("basket_code", as_index=False)
            .agg(
                avg_cash_weight=("cash_weight", "mean"),
                min_cash_weight=("cash_weight", "min"),
                max_cash_weight=("cash_weight", "max"),
                avg_turnover=("turnover", "mean"),
                avg_treasury_risk=("treasury_risk", "mean"),
                avg_broad_risk=("broad_risk", "mean"),
            )
            .sort_values("basket_code")
        )

    transition_summary = pd.DataFrame()
    if not tr.empty:
        transition_summary = (
            tr.groupby("basket_code", as_index=False)
            .agg(
                avg_turnover=("turnover", "mean"),
                avg_entries=("entries_n", "mean"),
                avg_exits=("exits_n", "mean"),
                avg_upsizes=("upsizes_n", "mean"),
                avg_downsizes=("downsizes_n", "mean"),
            )
            .sort_values("basket_code")
        )

    exposure_summary = pd.DataFrame()
    if not fx.empty:
        base_cols = [c for c in ["rebalance_date", "basket_code", "treasury_exposure_abs", "beta_SPY", "beta_QQQ", "beta_VIX", "beta_USO", "beta_GLD"] if c in fx.columns]
        exposure_summary = fx[base_cols].copy()

    ticker_summary = pd.DataFrame()
    if not tk.empty:
        ticker_summary = (
            tk.groupby(["domain", "basket_code", "basket_name"], as_index=False)
            .agg(
                selected_markets=("market_id", "nunique"),
                selected_tickers=("ticker_id", "nunique"),
                avg_chain_market_count=("chain_market_count", "mean"),
                max_chain_market_count=("chain_market_count", "max"),
                unique_event_families=("event_family_key", "nunique"),
            )
            .sort_values(["domain", "basket_code"])
        )

    lifecycle_summary = pd.DataFrame()
    if not lc.empty:
        ex = lc[lc["action"] == "EXIT"].copy()
        if not ex.empty:
            lifecycle_summary = (
                ex.groupby("basket_code", as_index=False)
                .agg(
                    exit_events=("market_id", "count"),
                    expiry_exits=("reason", lambda s: int((s == "expiry_or_resolution").sum())),
                    certainty_exits=("reason", lambda s: int((s == "certainty_threshold").sum())),
                    rebalance_exits=("reason", lambda s: int((s == "rebalance_replace").sum())),
                )
                .sort_values("basket_code")
            )

    cost_summary = pd.DataFrame()
    if not cm.empty:
        cost_summary = (
            cm.groupby("basket_code", as_index=False)
            .agg(
                avg_estimated_cost_bps=("estimated_total_cost_bps", "mean"),
                max_estimated_cost_bps=("estimated_total_cost_bps", "max"),
                avg_estimated_drag=("estimated_cost_drag_return", "mean"),
                avg_turnover=("turnover", "mean"),
                avg_cash_weight=("cash_weight", "mean"),
            )
            .sort_values("basket_code")
        )

    inception_summary = pd.DataFrame()
    if not ip.empty:
        inception_summary = ip[
            [
                "basket_code",
                "basket_name",
                "raw_history_start",
                "default_inception_date",
                "manual_override_date",
                "effective_inception_date",
                "price_coverage_at_inception",
                "cash_weight_at_inception",
                "rule_name",
                "override_reason",
            ]
        ].copy()

    exclusivity_qc = pd.DataFrame()
    if not compositions.empty and "exclusive_group_key" in compositions.columns:
        tmp = compositions[(~compositions["is_cash"]) & compositions["exclusive_group_key"].notna()].copy()
        tmp["exclusive_group_key"] = tmp["exclusive_group_key"].astype(str)
        tmp = tmp[tmp["exclusive_group_key"].str.len() > 0]
        if not tmp.empty:
            coll = (
                tmp.groupby(["rebalance_date", "basket_code", "exclusive_group_key"], as_index=False)["market_id"]
                .nunique()
                .rename(columns={"market_id": "n_contracts"})
            )
            coll = coll[coll["n_contracts"] > 1].copy()
            if not coll.empty:
                exclusivity_qc = coll.sort_values(["rebalance_date", "basket_code", "n_contracts"], ascending=[True, True, False]).head(40)

    certainty_examples = pd.DataFrame()
    expiry_examples = pd.DataFrame()
    if not lc.empty:
        certainty_examples = lc[lc["reason"] == "certainty_threshold"][
            ["rebalance_date", "basket_code", "market_id", "title", "price_at_prev", "reason"]
        ].head(25)
        expiry_examples = lc[lc["reason"] == "expiry_or_resolution"][
            ["rebalance_date", "basket_code", "market_id", "title", "end_date", "reason"]
        ].head(25)

    price_data_available = bool(
        (
            ("effective_price" in compositions.columns)
            and pd.to_numeric(compositions["effective_price"], errors="coerce").notna().any()
        )
        or (
            ("current_price" in compositions.columns)
            and pd.to_numeric(compositions["current_price"], errors="coerce").notna().any()
        )
    ) if not compositions.empty else False
    price_status = (
        "available (theme-aligned effective price and certainty rule active in selection diagnostics)"
        if price_data_available
        else "not available in this snapshot (certainty rule code is active but cannot fire without price input)"
    )
    agg_level_img = (
        f"<img src='{html.escape(aggregate_level_chart_rel)}' alt='aggregate level chart' style='width:100%;border:1px solid #e5e7eb;border-radius:8px;background:#fff;' />"
        if aggregate_level_chart_rel
        else "<p class='muted'>Aggregate level chart unavailable.</p>"
    )
    agg_cash_img = (
        f"<img src='{html.escape(aggregate_cash_chart_rel)}' alt='aggregate cash chart' style='width:100%;border:1px solid #e5e7eb;border-radius:8px;background:#fff;' />"
        if aggregate_cash_chart_rel
        else "<p class='muted'>Aggregate cash/turnover chart unavailable.</p>"
    )

    methodology_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Itô Markets | Methodology</title>
  <style>
    {_site_base_css("""
    .doc-layout {{
      margin-top: 14px;
    }}
    .doc-article p,
    .doc-article li {{
      color: var(--muted);
      line-height: 1.66;
      font-size: 14px;
    }}
    .doc-article ul {{
      margin-top: 6px;
      margin-bottom: 10px;
    }}
    .doc-article code {{
      background: #f3f6fb;
      border: 1px solid #e3e8f2;
      border-radius: 4px;
      padding: 1px 4px;
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .doc-article .lead {{
      font-size: 15px;
      color: #4b5563;
      margin-bottom: 8px;
    }}
    .img-wrap {{
      margin: 10px 0 14px;
    }}
    table.tbl {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      table-layout: fixed;
    }}
    table.tbl th, table.tbl td {{
      border-top: 1px solid #e5e7eb;
      padding: 7px 8px;
      vertical-align: top;
      word-break: break-word;
      text-align: left;
    }}
    table.tbl th {{
      background: #f9fafb;
    }}
    .small {{
      font-size: 12px;
      color: #6b7280;
    }}
    .formula {{
      margin: 8px 0 10px;
    }}
    .toc-panel li {{
      margin: 4px 0;
    }}
    """)}
  </style>
</head>
<body>
  {_site_nav("Methodology", "Research workflow and rulebook")}
    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <div class="eyebrow">Methodology</div>
          <h1>Complete rulebook for the static basket research workflow.</h1>
          <p>This methodology page is not a marketing summary. It is the operating document for the generated basket site: scope, data dependencies, selection logic, lifecycle handling, display rules, diagnostics, and the exact output artifacts used by the front end.</p>
          <div class="hero-actions">
            <span class="hero-chip"><strong>Coverage</strong> {html.escape(start)} → {html.escape(end)}</span>
            <span class="hero-chip"><strong>Baskets</strong> {n_baskets}</span>
            <span class="hero-chip"><strong>Contracts</strong> {n_contracts}</span>
            <span class="hero-chip"><strong>Rebalances</strong> {n_rebalances}</span>
          </div>
        </div>
        <div class="hero-profile">
          <div class="profile-card">
            <div class="section-kicker">Document Scope</div>
            <h3>Single workflow, single output system.</h3>
            <p>The methodology is aligned to the exact static artifacts generated by the engine in this workspace. That keeps the site, CSV exports, and written rulebook synchronized.</p>
            <div class="profile-grid">
              <div class="profile-row"><small>Focus</small><strong>Tradable NAV and basket construction</strong></div>
              <div class="profile-row"><small>Inputs</small><strong>Processed market and price data</strong></div>
              <div class="profile-row"><small>Outputs</small><strong>CSV artifacts + static website</strong></div>
              <div class="profile-row"><small>Mode</small><strong>Deterministic and reproducible</strong></div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <div class="doc-layout">
      <aside class="toc-panel">
        <div class="section-kicker">Contents</div>
        <h3>Navigation</h3>
        <p class="subtle">Use this as a fast index into the operating rulebook.</p>
        <div class="toc">
        <ul>
          <li>1. Scope and Design Intent</li>
          <li>2. Data Universe and Temporal Realism</li>
          <li>3. Candidate Filtering and Noise Rejection</li>
          <li>4. Selection Logic, Diversification, and Exclusivity</li>
          <li>5. Weighting, Cash, and Rebalance Mechanics</li>
          <li>6. Lifecycle, Expiry, and Certainty Handling</li>
          <li>7. Basket Inception and Display Rules</li>
          <li>8. Tradable NAV Publishing</li>
          <li>9. Directionality and Risk-Up Semantics</li>
          <li>10. Basket Slot Architecture</li>
          <li>11. Level Construction and Performance Proxies</li>
          <li>12. Governance Checks and Failure Modes</li>
          <li>13. Reproducible Output Artifacts</li>
          <li>14. Appendices (Full Tables)</li>
        </ul>
      </div>
      </aside>

      <article class="doc-article">
      <h2>1. Scope And Design Intent</h2>
      <p>The engine is intentionally single-purpose: construct thematic baskets that are coherent, tradable, and auditable. The strategy is rules-based with no manual override path during monthly construction. Every selection, exclusion, and weight comes from deterministic code and source data.</p>
      <p>The key design tradeoff is between thematic purity and execution realism. The process prioritizes pure theme exposure first, then applies implementability and risk controls (liquidity/history quality, diversification caps, turnover-aware cash sleeve, and contract lifecycle handling).</p>

      <h2>2. Data Universe And Temporal Realism</h2>
      <p>Universe data is loaded from <code>data/processed/ticker_mapping.parquet</code>. Contract chain lineage is loaded from <code>data/processed/ticker_chains.json</code>. Temporal listing controls are loaded from <code>data/processed/polymarket_market_history.parquet</code>. Pricing inputs come from <code>data/processed/prices.parquet</code> and <code>data/processed/returns.parquet</code>. The engine may use optional precomputed enrichment columns when they already exist inside the processed market universe, but it does not depend on any separate legacy clustering pipeline.</p>
      <p>Temporal realism is enforced with two rules. First, at each rebalance date, contracts must satisfy minimum time-to-expiry. Second, if inferred first-seen information is available, contracts are excluded before their inferred market appearance date. This prevents lookahead contamination when historical snapshots are reconstructed from cumulative data dumps.</p>
      <ul>
        <li>Minimum days-to-expiry threshold: <code>21</code> days.</li>
        <li>Ticker-chain de-duplication: one selected market per <code>ticker_id</code> at a rebalance.</li>
        <li>Certainty-input status in this snapshot: <strong>{html.escape(price_status)}</strong>.</li>
      </ul>

      <h2>3. Candidate Filtering And Noise Rejection</h2>
      <p>Filtering is applied in layers to remove non-semantic noise before scoring. Global filters remove election campaign contracts and broad sports/entertainment contracts. Basket-specific filters then remove terms that introduce off-theme chatter. Category gating keeps contracts aligned with each basket taxonomy, while still allowing limited high-keyword fallback from unknown categories when needed for coverage.</p>
      <p>This layered gate is the primary anti-noise control: contracts must pass theme relevance, category compatibility, and hard exclusions before they can compete for selection.</p>
      <ul>
        <li>Global hard rejects: election/presidential/campaign and broad sports/entertainment classes.</li>
        <li>Basket-level required patterns and exclusion patterns.</li>
        <li>Optional certainty filter when price data exists: remove contracts with theme-aligned effective price ≤ 0.05 or ≥ 0.95.</li>
      </ul>

      <h2>4. Selection Logic, Diversification, And Exclusivity</h2>
      <p>After filtering, each candidate receives a multi-factor selection score that blends thematic relevance with quality and stability. The selector then runs a greedy marginal-add process, where each new contract must improve basket objective net of similarity and concentration penalties.</p>
      <div class="formula">
        <p><code>selection_score = 0.36*keyword_z + 0.27*quality_z + 0.10*stability_z + 0.05*trend_z + 0.08*category_z</code></p>
        <p><code>marginal = selection_score - 0.35*max_text_similarity - template_penalty - event_penalty - exclusivity_penalty - community_penalty - factor_penalty</code></p>
      </div>
      <p>Diversification controls enforce limits at multiple semantic levels: template, community, event-family, and explicit mutual-exclusion groups. The mutual-exclusion layer is critical for ranking-style markets (for example, “best/top/second/third AI model”): contracts sharing the same exclusivity key cannot stack in one basket-month.</p>
      <p>The cap vector for each basket is codified in configuration and applied mechanically at every rebalance; no discretionary exception path exists.</p>

      <h2>5. Weighting, Cash, And Rebalance Mechanics</h2>
      <p>Selected contracts are ranked by score, quality, liquidity, and stability to produce raw weights, then clipped and normalized under basket concentration caps. The investable sleeve equals <code>1 - cash</code>. Cash is dynamic, not fixed, and rises under turnover and risk stress.</p>
      <div class="formula">
        <p><code>raw_weight = 0.25 + 0.45*rank(selection_score) + 0.22*rank(quality) + 0.18*rank(log(1+volume)) + 0.15*rank(stability)</code></p>
      </div>
      <p><strong>Cash is not transaction cost.</strong> Cash is reserve allocation inside basket construction. Cost is estimated separately using turnover and expiry/certainty pressure proxies. Rebalance occurs monthly using only information available at rebalance time.</p>

      <h2>6. Lifecycle, Expiry, And Certainty Handling</h2>
      <p>Contracts naturally roll out for expiry/resolution and are replaced by current best candidates at subsequent rebalance. If price data is available, certainty-threshold exits (≥95% or ≤5%) are enforced on the held side of the contract: YES uses the raw YES price, while NO uses <code>1 - YES price</code>. Proceeds remain in cash until reallocation through the normal rebalance logic.</p>
      <p>Lifecycle diagnostics are exported as an event stream so every entry/exit has a machine-readable reason: new selection, expiry/resolution, certainty threshold, or rebalance replacement.</p>
      <div class="formula">
        <p><code>estimated_cost_bps = 18 + 35*turnover + 10*expiring_30d_share + 6*expiring_60d_share + 25*certainty_share</code></p>
        <p class="small">Certainty component is active only when price inputs are present.</p>
      </div>

      <h2>7. Basket Inception And Display Rules</h2>
      <p>The stored basket history can begin before a basket is meaningfully investable. In those periods the basket is effectively just cash or low-coverage placeholder exposure, which produces long flat chart segments that are technically present in the raw data but economically misleading as a default view.</p>
      <p>The dashboard therefore computes an effective inception date for each basket. The automatic rule is the first row where <code>price_coverage_share ≥ 0.60</code> and <code>cash_weight ≤ 0.25</code>. If no row satisfies that rule, the fallback is the first date where <code>cash_weight &lt; 0.95</code>. If a manual override exists in <code>config/basket_inception_overrides.yml</code>, that override becomes the effective inception date. Full raw history remains accessible in the website through a Full History mode.</p>
      {_table_html(inception_summary, max_rows=50)}

      <h2>8. Tradable NAV Publishing</h2>
      <p>The published level is now just <strong>Tradable NAV</strong>. It answers one question: “what money would the buyer have made by owning the basket through rolls, exits, costs, and cash management?”</p>
      <p>For baskets like <code>ADIT-S3</code>, this keeps the chart focused on the tradable object rather than mixing it with a separate normalized risk overlay. The remaining diagnostics on the page are there to show coverage, concentration, and composition quality, not to publish a second headline index.</p>
      <p>Tenor pressure has been removed from both selection scoring and maintenance. The engine now keeps only a minimal operational roll near expiry instead of forcing the basket toward a target tenor or expiry preference profile.</p>

      <h2>9. Directionality And Risk-Up Semantics</h2>
      <p>Every non-cash position is directionally explicit. The product distinguishes the raw market <code>YES</code> probability from the basket’s <code>Risk Price</code>. If the basket is long a risk-up escalation market, it holds <code>Long YES</code> and the risk price equals market YES. If the basket is long a de-escalation market on the opposite side, it holds <code>Long NO</code> and the risk price equals <code>1 - Market YES</code>.</p>
      <ul>
        <li><code>Long YES</code>: risk rises when the market YES probability rises.</li>
        <li><code>Long NO</code>: risk rises when the market YES probability falls because the market itself describes de-escalation, normalization, or peace.</li>
        <li>Website contract tables therefore show <code>Instruction</code>, <code>Market YES</code>, <code>Risk Price</code>, and a short <code>Why</code> label.</li>
      </ul>
      <p>Example: if a contract asks “Will Israel and Saudi Arabia normalize relations by March 31?”, the basket may hold <code>Long NO</code>. In that case the raw market YES can be low while the basket’s risk price is high, because failed normalization is treated as risk-up for the basket.</p>

      <h2>10. Basket Slot Architecture</h2>
      <p>Broad pattern matching alone was not strong enough for the highest-sensitivity geopolitical and energy baskets, so the engine now supports explicit slot schemas. Slot schemas are hard-coded thematic sub-buckets with per-slot caps, required-if-available rules, and proxy controls.</p>
      <ul>
        <li><code>ADIT-S3</code>: Iran-Israel escalation, Hamas ceasefire failure, Hezbollah/Lebanon, Red Sea/Hormuz disruption, US-Iran intervention, normalization failure.</li>
        <li><code>ADIT-S2</code>: Russia territorial advance, ceasefire failure, NATO direct clash, Greenland/Arctic/Nordic tension, nuclear/strategic escalation, EU/NATO security fracture.</li>
        <li><code>ADIT-E3</code>: Hormuz/Red Sea disruption, export-terminal/refinery hits, OPEC restriction, shipping disruption, producer shortfall, and capped oil-price proxy slots.</li>
      </ul>
      <p>This slot architecture is what prevents <code>ADIT-S2</code> from degenerating into a pile of near-zero tail contracts, and what prevents <code>ADIT-E3</code> from becoming an empty or pure crude-ladder basket when direct disruption markets exist.</p>

      <h2>11. Level Construction And Performance Proxies</h2>
      <p>Two level paths are published. First, per-basket monthly level proxies from weighted contract prices plus cash sleeve. Second, an aggregate weighted index-level path across baskets. Each path is aligned with rebalance points and accompanied by entries/exits and turnover diagnostics.</p>
      <div class="img-wrap">
        {agg_level_img}
      </div>
      <div class="img-wrap">
        {agg_cash_img}
      </div>
      <p>Macro-factor overlays are computed from market-level beta exposures inherited from community factor models and aggregated by basket weights each month.</p>

      <h2>12. Governance Checks And Failure Modes</h2>
      <p>Quality control checks focus on coherence and leak prevention: no stale-expiry holdings after rebalance, no lookahead against inferred first-seen dates, no duplicate exclusivity groups per basket-month, and bounded position/cash constraints. If any check fails, artifacts make the failure visible at CSV level for audit and correction.</p>

      <h2>13. Reproducible Output Artifacts</h2>
      <p>The following outputs define the single operational workflow and are sufficient for third-party replication:</p>
      <ul>
        <li><code>last_year_monthly_compositions.csv</code>: full contract-level composition each month.</li>
        <li><code>basket_level_monthly.csv</code>: per-basket monthly level, cash, entries/exits, turnover.</li>
        <li><code>aggregate_basket_level.csv</code>: aggregate level path with rebalance-change diagnostics.</li>
        <li><code>basket_inception_policy.csv</code>: raw history start, automatic inception date, override, and effective dashboard start for each basket.</li>
        <li><code>slot_coverage_diagnostics.csv</code>: per-slot candidate and selected coverage by rebalance.</li>
        <li><code>monthly_cash_positions.csv</code>: monthly cash and risk context by basket.</li>
        <li><code>contract_lifecycle_events.csv</code>: entry/exit events with reasons.</li>
        <li><code>rebalance_transitions.csv</code> and <code>rebalance_cost_model.csv</code>: execution diagnostics.</li>
      </ul>

      <h2>14. Appendices (Full Tables)</h2>
      <details>
        <summary>Appendix A: Full Basket Rulebook</summary>
        <table class="tbl">
          <thead>
            <tr>
              <th>Domain</th><th>Code</th><th>Basket</th><th>Target N</th><th>Min-Max N</th><th>Cash</th><th>Allowed Categories</th><th>Diversification Caps</th><th>Include Patterns</th><th>Exclude Patterns</th>
            </tr>
          </thead>
          <tbody>
            {"".join(rows)}
          </tbody>
        </table>
      </details>
      <details>
        <summary>Appendix B: Ticker Coverage Summary</summary>
        {_table_html(ticker_summary, max_rows=60)}
      </details>
      <details>
        <summary>Appendix C: Aggregate Time Series</summary>
        {_table_html(ag, max_rows=180)}
      </details>
      <details>
        <summary>Appendix D: Cash, Cost, Transition, Lifecycle, Exclusivity QC</summary>
        <h3>Cash Policy Summary</h3>
        {_table_html(cash_summary, max_rows=60)}
        <h3>Cost Summary</h3>
        {_table_html(cost_summary, max_rows=60)}
        <h3>Transition Summary</h3>
        {_table_html(transition_summary, max_rows=60)}
        <h3>Lifecycle Exit Summary</h3>
        {_table_html(lifecycle_summary, max_rows=60)}
        <h3>Mutual-Exclusion QC (Should Be Empty)</h3>
        {_table_html(exclusivity_qc, max_rows=60)}
      </details>
      <details>
        <summary>Appendix E: Certainty and Expiry Exit Examples</summary>
        <h3>Certainty-Threshold Exit Examples (95/5 Rule)</h3>
        {_table_html(certainty_examples, max_rows=40)}
        <h3>Expiry/Resolution Exit Examples</h3>
        {_table_html(expiry_examples, max_rows=40)}
      </details>
      <details>
        <summary>Appendix F: Factor Exposure Snapshot</summary>
        {_table_html(exposure_summary, max_rows=120)}
      </details>
    </article>
  </div>
</body>
</html>
"""
    output_path.write_text(methodology_html, encoding="utf-8")


def _build_website_html(
    site_dir: Path,
    summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    compositions: pd.DataFrame,
    specs: list[ThematicBasketSpec],
    start: str,
    end: str,
    ticker_chain_meta: pd.DataFrame | None = None,
    ticker_chain_history: pd.DataFrame | None = None,
    basket_level_series: pd.DataFrame | None = None,
    aggregate_series: pd.DataFrame | None = None,
    cash_positions: pd.DataFrame | None = None,
    transitions: pd.DataFrame | None = None,
    factor_exposure: pd.DataFrame | None = None,
    lifecycle_events: pd.DataFrame | None = None,
    cost_model: pd.DataFrame | None = None,
    inception_policy: pd.DataFrame | None = None,
) -> None:
    site_dir.mkdir(parents=True, exist_ok=True)
    asset_paths = _copy_site_assets(site_dir)
    charts_dir = site_dir / "charts"
    baskets_charts = charts_dir / "baskets"
    contracts_charts = charts_dir / "contracts"
    summary_charts = charts_dir / "summary"
    baskets_charts.mkdir(parents=True, exist_ok=True)
    contracts_charts.mkdir(parents=True, exist_ok=True)
    summary_charts.mkdir(parents=True, exist_ok=True)

    c = compositions.copy()
    c["rebalance_date"] = pd.to_datetime(c["rebalance_date"])
    c["end_date"] = pd.to_datetime(c["end_date"], errors="coerce")
    non_cash = c[~c["is_cash"]].copy()

    bl = basket_level_series.copy() if isinstance(basket_level_series, pd.DataFrame) else pd.DataFrame()
    if bl.empty:
        bl = _build_basket_level_series(c, transitions if isinstance(transitions, pd.DataFrame) else None)
    if not bl.empty:
        bl["rebalance_date"] = pd.to_datetime(bl["rebalance_date"], errors="coerce")
        bl = bl.sort_values(["basket_code", "rebalance_date"]).copy()
    bl_latest_map: dict[str, dict[str, object]] = {}
    if not bl.empty:
        for code, g in bl.groupby("basket_code", sort=True):
            last = g.sort_values("rebalance_date").iloc[-1]
            bl_latest_map[str(code)] = {
                "tradable_nav": None if pd.isna(last.get("tradable_nav")) else float(last.get("tradable_nav")),
                "cash_weight": None if pd.isna(last.get("cash_weight")) else float(last.get("cash_weight")),
                "price_coverage_share": None if pd.isna(last.get("price_coverage_share")) else float(last.get("price_coverage_share")),
                "spot_risk_level": None if pd.isna(last.get("spot_risk_level")) else float(last.get("spot_risk_level")),
                "spot_weighted_effective_price_horizon": None if pd.isna(last.get("spot_weighted_effective_price_horizon")) else float(last.get("spot_weighted_effective_price_horizon")),
                "spot_weighted_dte_days": None if pd.isna(last.get("spot_weighted_dte_days")) else float(last.get("spot_weighted_dte_days")),
                "spot_price_coverage_share": None if pd.isna(last.get("spot_price_coverage_share")) else float(last.get("spot_price_coverage_share")),
                "spot_stale_weight_share": None if pd.isna(last.get("spot_stale_weight_share")) else float(last.get("spot_stale_weight_share")),
                "nav_spot_gap": None if pd.isna(last.get("nav_spot_gap")) else float(last.get("nav_spot_gap")),
                "tenor_target_days": None if pd.isna(last.get("tenor_target_days")) else float(last.get("tenor_target_days")),
                "tenor_drift_days": None if pd.isna(last.get("tenor_drift_days")) else float(last.get("tenor_drift_days")),
                "tail_probability_weight_share": None if pd.isna(last.get("tail_probability_weight_share")) else float(last.get("tail_probability_weight_share")),
                "proxy_weight_share": None if pd.isna(last.get("proxy_weight_share")) else float(last.get("proxy_weight_share")),
                "slot_coverage_ratio": None if pd.isna(last.get("slot_coverage_ratio")) else float(last.get("slot_coverage_ratio")),
                "direction_balance_score": None if pd.isna(last.get("direction_balance_score")) else float(last.get("direction_balance_score")),
            }

    inception_df = inception_policy.copy() if isinstance(inception_policy, pd.DataFrame) else pd.DataFrame()
    if inception_df.empty:
        inception_df = _build_inception_policy(bl, specs, _load_inception_overrides(specs))
    inception_map = {
        str(r["basket_code"]): {
            "raw_history_start": str(r.get("raw_history_start", "") or ""),
            "default_inception_date": str(r.get("default_inception_date", "") or ""),
            "manual_override_date": str(r.get("manual_override_date", "") or ""),
            "effective_inception_date": str(r.get("effective_inception_date", "") or ""),
            "price_coverage_at_inception": None if pd.isna(r.get("price_coverage_at_inception")) else float(r.get("price_coverage_at_inception")),
            "cash_weight_at_inception": None if pd.isna(r.get("cash_weight_at_inception")) else float(r.get("cash_weight_at_inception")),
            "rule_name": str(r.get("rule_name", "") or ""),
            "override_reason": str(r.get("override_reason", "") or ""),
        }
        for _, r in inception_df.iterrows()
    }

    agg = aggregate_series.copy() if isinstance(aggregate_series, pd.DataFrame) else pd.DataFrame()
    if agg.empty and not c.empty:
        agg = _build_aggregate_basket_series(c, transitions if isinstance(transitions, pd.DataFrame) else None)
    agg_level_rel = "charts/summary/aggregate_basket_level.png"
    agg_cash_rel = "charts/summary/aggregate_cash_turnover.png"
    if not agg.empty:
        _render_aggregate_level_chart(agg, summary_charts / "aggregate_basket_level.png")
        _render_aggregate_cash_chart(agg, summary_charts / "aggregate_cash_turnover.png")

    baskets_page_details: dict[str, dict] = {}
    month_points = sorted(non_cash["rebalance_date"].dropna().unique())
    month_labels = [pd.Timestamp(x).date().isoformat() for x in month_points]
    basket_order = sorted(non_cash["basket_code"].unique())

    chain_meta_map: dict[str, dict] = {}
    if isinstance(ticker_chain_meta, pd.DataFrame) and not ticker_chain_meta.empty:
        cm = ticker_chain_meta.copy().drop_duplicates("ticker_id", keep="first")
        for _, r in cm.iterrows():
            tid = str(r.get("ticker_id", ""))
            chain_meta_map[tid] = {
                "ticker_name": str(r.get("ticker_name", "")),
                "chain_market_count": int(r.get("chain_market_count", 0) or 0),
                "chain_first_end_date": str(r.get("chain_first_end_date", "")),
                "chain_last_end_date": str(r.get("chain_last_end_date", "")),
            }

    chain_history_map: dict[str, list[dict]] = {}
    if isinstance(ticker_chain_history, pd.DataFrame) and not ticker_chain_history.empty:
        ch = ticker_chain_history.copy()
        for tid, g in ch.groupby("ticker_id", sort=True):
            rows: list[dict] = []
            for _, r in g.sort_values(["chain_order", "end_date", "market_id"]).iterrows():
                rows.append(
                    {
                        "market_id": str(r.get("market_id", "")),
                        "title": str(r.get("title", "")),
                        "end_date": "" if pd.isna(r.get("end_date")) else pd.Timestamp(r.get("end_date")).date().isoformat(),
                        "event_slug": str(r.get("event_slug", "")),
                        "selected_any": bool(r.get("selected_any", False)),
                        "selected_rebalance_dates": str(r.get("selected_rebalance_dates", "")),
                    }
                )
            chain_history_map[str(tid)] = rows

    def _last_string(frame: pd.DataFrame, column: str, default: str = "") -> str:
        if column not in frame.columns:
            return default
        series = frame[column].dropna()
        if series.empty:
            return default
        return str(series.iloc[-1])

    def _last_float(frame: pd.DataFrame, column: str) -> float | None:
        if column not in frame.columns:
            return None
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            return None
        return float(series.iloc[-1])

    explorer_baskets: list[dict] = []
    explorer_contracts_by_basket: dict[str, dict] = {}

    for code in basket_order:
        bg = non_cash[non_cash["basket_code"] == code].copy()
        if bg.empty:
            continue
        meta = {
            "code": code,
            "name": str(bg["basket_name"].iloc[0]),
            "domain": str(bg["domain"].iloc[0]),
            "basket_weight": float(bg["basket_weight"].iloc[0]),
            "cash_weight": float(c[(c["basket_code"] == code) & (c["is_cash"])]["target_weight"].iloc[0]),
            "effective_inception_date": str(inception_map.get(code, {}).get("effective_inception_date", "")),
            **bl_latest_map.get(code, {}),
        }
        explorer_baskets.append(meta)

        months_map: dict[str, list[dict]] = {}
        for d, g in bg.groupby("rebalance_date", sort=True):
            d_iso = pd.Timestamp(d).date().isoformat()
            month_rows: list[dict] = []
            for _, r in g.sort_values("target_weight", ascending=False).iterrows():
                expiry = "" if pd.isna(r["end_date"]) else pd.Timestamp(r["end_date"]).date().isoformat()
                tid = str(r["ticker_id"])
                cmeta = chain_meta_map.get(tid, {})
                month_rows.append(
                    {
                        "market_id": str(r["market_id"]),
                        "title": str(r["title"]),
                        "weight": float(r["target_weight"]),
                        "end_date": expiry,
                        "ticker_id": tid,
                        "ticker_name": str(cmeta.get("ticker_name", "")),
                        "chain_market_count": int(cmeta.get("chain_market_count", 0) or 0),
                        "position_instruction": str(r.get("position_instruction", _make_position_instruction(r.get("position_side")))),
                        "market_yes_price": None if pd.isna(r.get("market_yes_price")) else float(r.get("market_yes_price")),
                        "effective_risk_price": None if pd.isna(r.get("effective_risk_price")) else float(r.get("effective_risk_price")),
                        "spot_effective_price_horizon": None if pd.isna(r.get("spot_effective_price_horizon")) else float(r.get("spot_effective_price_horizon")),
                        "direction_reason": str(r.get("direction_reason", "")),
                        "slot_name": str(r.get("slot_name", "")),
                        "tenor_band_status": str(r.get("tenor_band_status", "")),
                        "days_to_expiry": None if pd.isna(r.get("days_to_expiry")) else float(r.get("days_to_expiry")),
                    }
                )
            months_map[d_iso] = month_rows

        contracts_map: dict[str, dict] = {}
        for market_id, cg in bg.groupby("market_id", sort=True):
            weights = {m: 0.0 for m in month_labels}
            for _, r in cg.iterrows():
                m = pd.Timestamp(r["rebalance_date"]).date().isoformat()
                weights[m] = float(r["target_weight"])
            active_months = [m for m in month_labels if float(weights.get(m, 0.0)) > 0]
            expiry = ""
            if cg["end_date"].notna().any():
                expiry = pd.Timestamp(cg["end_date"].dropna().max()).date().isoformat()
            contracts_map[str(market_id)] = {
                "market_id": str(market_id),
                "title": str(cg["title"].iloc[0]),
                "ticker_id": str(cg["ticker_id"].iloc[0]),
                "end_date": expiry,
                "avg_weight": float(cg["target_weight"].mean()),
                "peak_weight": float(cg["target_weight"].max()),
                "weights": weights,
                "first_active_month": active_months[0] if active_months else "",
                "last_active_month": active_months[-1] if active_months else "",
                "ticker_name": str(chain_meta_map.get(str(cg["ticker_id"].iloc[0]), {}).get("ticker_name", "")),
                "chain_market_count": int(chain_meta_map.get(str(cg["ticker_id"].iloc[0]), {}).get("chain_market_count", 0) or 0),
                "chain_first_end_date": str(chain_meta_map.get(str(cg["ticker_id"].iloc[0]), {}).get("chain_first_end_date", "")),
                "chain_last_end_date": str(chain_meta_map.get(str(cg["ticker_id"].iloc[0]), {}).get("chain_last_end_date", "")),
                "chain_markets": chain_history_map.get(str(cg["ticker_id"].iloc[0]), []),
                "position_instruction": _last_string(cg, "position_instruction", _make_position_instruction(cg["position_side"].iloc[0]) if "position_side" in cg.columns else "LONG_YES"),
                "market_yes_price": _last_float(cg, "market_yes_price"),
                "effective_risk_price": _last_float(cg, "effective_risk_price"),
                "spot_effective_price_horizon": _last_float(cg, "spot_effective_price_horizon"),
                "direction_reason": _last_string(cg, "direction_reason", ""),
                "slot_name": _last_string(cg, "slot_name", ""),
                "tenor_band_status": _last_string(cg, "tenor_band_status", ""),
            }

        explorer_contracts_by_basket[code] = {
            "meta": meta,
            "months": months_map,
            "contracts": contracts_map,
        }

    explorer_payload = {
        "months": month_labels,
        "baskets": explorer_baskets,
        "contracts_by_basket": explorer_contracts_by_basket,
    }
    explorer_payload_json = json.dumps(explorer_payload, ensure_ascii=False).replace("</", "<\\/")

    for code in basket_order:
        bg = non_cash[non_cash["basket_code"] == code].copy()
        if bg.empty:
            continue
        basket_bundle = explorer_contracts_by_basket.get(code, {})
        basket_meta = dict(basket_bundle.get("meta", {}))
        basket_contracts = dict(basket_bundle.get("contracts", {}))
        if not basket_meta:
            basket_meta = {
                "code": code,
                "name": str(bg["basket_name"].iloc[0]),
                "domain": str(bg["domain"].iloc[0]),
                "basket_weight": float(bg["basket_weight"].iloc[0]),
                "cash_weight": float(c[(c["basket_code"] == code) & (c["is_cash"])]["target_weight"].iloc[0]),
                "effective_inception_date": str(inception_map.get(code, {}).get("effective_inception_date", "")),
            }

        basket_chart_name = f"{_slug(code)}_composition.png"
        basket_chart_rel = f"charts/baskets/{basket_chart_name}"
        basket_chart_path = baskets_charts / basket_chart_name
        if not basket_chart_path.exists():
            _render_basket_composition_chart(bg, str(basket_meta.get("name", code)), basket_chart_path)

        ms = monthly_summary[monthly_summary["basket_code"] == code].copy()

        baskets_page_details[code] = {
            "meta": {
                **basket_meta,
                "unique_contracts": int(bg["market_id"].nunique()),
                "composition_chart": basket_chart_rel,
                "warning_tail_heavy": bool((basket_meta.get("tail_probability_weight_share") or 0) >= 0.40),
                "warning_low_coverage": bool((basket_meta.get("price_coverage_share") or 1) < 0.65),
                "warning_missing_slots": bool((basket_meta.get("slot_coverage_ratio") or 1) < 0.75),
                "warning_proxy_dominated": bool((basket_meta.get("proxy_weight_share") or 0) > 0.35),
            },
            "monthly_summary_rows": [
                {
                    "rebalance_date": html.escape(str(row["rebalance_date"])[:10]),
                    "n_contracts": int(row["n_contracts"]),
                    "avg_contract_weight": float(row["avg_contract_weight"]),
                    "max_contract_weight": float(row["max_contract_weight"]),
                }
                for _, row in ms.sort_values("rebalance_date").iterrows()
            ],
            "contracts": basket_contracts,
        }

    baskets_page_payload_json = json.dumps(
        {
            "months": month_labels,
            "baskets": explorer_baskets,
            "details": baskets_page_details,
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")

    summary_table = summary.copy()
    summary_table["basket_weight"] = summary_table["basket_weight"].map(lambda x: f"{float(x):.4f}")
    summary_table["cash_weight"] = summary_table["cash_weight"].map(lambda x: f"{float(x):.2%}")
    summary_table["avg_contract_weight"] = summary_table["avg_contract_weight"].map(lambda x: f"{float(x):.4f}")
    summary_table["avg_days_to_expiry"] = summary_table["avg_days_to_expiry"].map(lambda x: f"{float(x):.1f}")
    summary_table_html = summary_table.to_html(index=False, classes=["summary-table"])

    basket_level_payload: dict[str, dict] = {}
    if not bl.empty:
        for code, g in bl.groupby("basket_code", sort=True):
            g = g.sort_values("rebalance_date")
            gap_overlay = _build_stochastic_gap_overlay(g, str(code))
            inception_meta = inception_map.get(str(code), {})
            basket_level_payload[str(code)] = {
                "code": str(code),
                "domain": str(g["domain"].iloc[0]) if "domain" in g.columns and len(g) else "",
                "basket_name": str(g["basket_name"].iloc[0]) if "basket_name" in g.columns and len(g) else str(code),
                "raw_history_start": inception_meta.get("raw_history_start", ""),
                "default_inception_date": inception_meta.get("default_inception_date", ""),
                "manual_override_date": inception_meta.get("manual_override_date", ""),
                "effective_inception_date": inception_meta.get("effective_inception_date", ""),
                "price_coverage_at_inception": inception_meta.get("price_coverage_at_inception", None),
                "cash_weight_at_inception": inception_meta.get("cash_weight_at_inception", None),
                "inception_rule": inception_meta.get("rule_name", ""),
                "rows": [
                    {
                        "rebalance_date": pd.Timestamp(r["rebalance_date"]).date().isoformat(),
                        "tradable_nav": float(r.get("tradable_nav", r.get("basket_level", np.nan))),
                        "basket_level": float(r.get("basket_level", np.nan)),
                        "spot_risk_level": None if pd.isna(r.get("spot_risk_level")) else float(r.get("spot_risk_level")),
                        "spot_weighted_effective_price_raw": None if pd.isna(r.get("spot_weighted_effective_price_raw")) else float(r.get("spot_weighted_effective_price_raw")),
                        "spot_weighted_effective_price_horizon": None if pd.isna(r.get("spot_weighted_effective_price_horizon")) else float(r.get("spot_weighted_effective_price_horizon")),
                        "spot_weighted_dte_days": None if pd.isna(r.get("spot_weighted_dte_days")) else float(r.get("spot_weighted_dte_days")),
                        "spot_price_coverage_share": None if pd.isna(r.get("spot_price_coverage_share")) else float(r.get("spot_price_coverage_share")),
                        "spot_stale_weight_share": None if pd.isna(r.get("spot_stale_weight_share")) else float(r.get("spot_stale_weight_share")),
                        "nav_spot_gap": None if pd.isna(r.get("nav_spot_gap")) else float(r.get("nav_spot_gap")),
                        "tenor_target_days": None if pd.isna(r.get("tenor_target_days")) else float(r.get("tenor_target_days")),
                        "tenor_drift_days": None if pd.isna(r.get("tenor_drift_days")) else float(r.get("tenor_drift_days")),
                        "tail_probability_weight_share": None if pd.isna(r.get("tail_probability_weight_share")) else float(r.get("tail_probability_weight_share")),
                        "proxy_weight_share": None if pd.isna(r.get("proxy_weight_share")) else float(r.get("proxy_weight_share")),
                        "slot_coverage_ratio": None if pd.isna(r.get("slot_coverage_ratio")) else float(r.get("slot_coverage_ratio")),
                        "direction_balance_score": None if pd.isna(r.get("direction_balance_score")) else float(r.get("direction_balance_score")),
                        "price_coverage_share": None if pd.isna(r.get("price_coverage_share")) else float(r.get("price_coverage_share")),
                        "cash_weight": float(r.get("cash_weight", np.nan)),
                        "entries": int(r.get("entries", 0)),
                        "exits": int(r.get("exits", 0)),
                        "turnover": float(r.get("turnover", 0.0)),
                        "rebalanced": bool(r.get("rebalanced", False)),
                        "graph_overlay_nav": None if gap_overlay[idx] is None else float(gap_overlay[idx]),
                    }
                    for idx, (_, r) in enumerate(g.iterrows())
                ],
            }
    basket_order_payload = [str(x) for x in summary["basket_code"].astype(str).tolist()] if not summary.empty else sorted(basket_level_payload.keys())
    basket_level_payload_json = json.dumps(
        {"order": basket_order_payload, "series": basket_level_payload},
        ensure_ascii=False,
    ).replace("</", "<\\/")

    index_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Itô Markets | Basket Dashboard</title>
  <style>
    {_site_base_css("""
    .overview-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(340px, 0.85fr);
      gap: 14px;
      align-items: start;
    }}
    .table-card {{
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      background: #fff;
    }}
    .table-card .table-inner {{
      overflow: auto;
      max-height: 520px;
    }}
    .chart-stack {{
      display: grid;
      gap: 12px;
    }}
    .dashboard-shell {{
      display: grid;
      gap: 14px;
    }}
    .dashboard-hero-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
      gap: 14px;
      align-items: stretch;
    }}
    .basket-hero-panel {{
      padding: 20px 22px;
      border: 1px solid var(--line);
      border-radius: 22px;
      background: linear-gradient(180deg, #ffffff 0%, #f7faff 100%);
      box-shadow: var(--shadow);
    }}
    .basket-hero-panel h2 {{
      margin-bottom: 8px;
      font-size: clamp(26px, 3vw, 38px);
    }}
    .hero-level {{
      font-size: clamp(54px, 7vw, 82px);
      line-height: 0.9;
      letter-spacing: -0.05em;
      font-weight: 700;
      margin: 16px 0 8px;
    }}
    .hero-change {{
      display: flex;
      align-items: baseline;
      gap: 10px;
      font-size: 24px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .hero-change span {{
      font-size: 16px;
      font-weight: 700;
    }}
    .hero-change.up {{
      color: var(--success);
    }}
    .hero-change.down {{
      color: var(--danger);
    }}
    .hero-asof {{
      margin-top: 8px;
      color: var(--muted-soft);
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      font-weight: 700;
    }}
    .hero-copyline {{
      margin-top: 14px;
      font-size: 14px;
      max-width: 760px;
    }}
    .path-toolbar {{
      display: grid;
      grid-template-columns: minmax(320px, 1.25fr) repeat(3, minmax(180px, 0.8fr)) minmax(260px, 1fr);
      gap: 12px;
      align-items: end;
    }}
    .basket-picker {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto auto;
      gap: 8px;
    }}
    .segmented {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
    }}
    .range-buttons {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 6px;
    }}
    .segmented button,
    .range-buttons button {{
      background: #f7f9fc;
    }}
    .segmented button.active,
    .range-buttons button.active {{
      background: linear-gradient(180deg, #1a4faa 0%, #173f86 100%);
      color: #fff;
      border-color: #173f86;
      box-shadow: 0 10px 20px rgba(26, 79, 170, 0.20);
    }}
    .toggle-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      padding: 11px 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #f7f9fc;
      min-height: 44px;
    }}
    .check {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text);
      font-size: 13px;
      font-weight: 700;
    }}
    .check input {{
      margin: 0;
    }}
    #basketPathChart {{
      min-height: 560px;
    }}
    .summary-caption {{
      margin-top: 8px;
      font-size: 13px;
    }}
    @media (max-width: 1220px) {{
      .overview-grid,
      .dashboard-hero-grid,
      .path-toolbar {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 760px) {{
      .basket-picker,
      .range-buttons,
      .segmented,
      .toggle-grid {{
        grid-template-columns: 1fr;
      }}
      .hero-level {{
        font-size: 52px;
      }}
    }}
    """)}
  </style>
  <script src="{asset_paths['echarts_js']}"></script>
</head>
<body>
  {_site_nav("Dashboard", "Prediction-market research terminal")}
    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <div class="eyebrow">Thematic Prediction Indices</div>
          <h1>Static research surface for systematic basket analysis.</h1>
          <p>The dashboard is generated directly from the basket engine outputs and is meant to behave like a front-end research terminal: clear defaults, strong basket profiles, direct links into contract archives, and a methodology trail that stays tied to the exact static artifacts on disk.</p>
          <div class="hero-actions">
            <span class="hero-chip"><strong>Window</strong> {html.escape(start)} → {html.escape(end)}</span>
            <span class="hero-chip"><strong>Baskets</strong> {int(summary["basket_code"].nunique()) if not summary.empty else 0}</span>
            <span class="hero-chip"><strong>Contracts</strong> {int(non_cash["market_id"].nunique()) if not non_cash.empty else 0}</span>
            <span class="hero-chip"><strong>Mode</strong> Tradable NAV focus</span>
          </div>
        </div>
        <div class="hero-profile">
          <div class="profile-card">
            <div class="section-kicker">Research Modes</div>
            <h3>One surface, three audit paths.</h3>
            <p>Use the dashboard for basket paths, the explorer for month-by-month contract selection, and the baskets archive for continuous contract composition history. The methodology page stays in the same shell so the analytical view and rulebook never drift.</p>
            <div class="profile-grid">
              <div class="profile-row"><small>Dashboard</small><strong>Level path + coverage + warnings</strong></div>
              <div class="profile-row"><small>Explorer</small><strong>Month-specific contract inspection</strong></div>
              <div class="profile-row"><small>Baskets</small><strong>Continuous contract archive</strong></div>
              <div class="profile-row"><small>Methodology</small><strong>Reproducible rulebook and output map</strong></div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <div class="section-kicker">System Overview</div>
          <h2>Current basket roster and aggregate path context.</h2>
          <p>The table below is the active index roster generated by the current run. The aggregate charts remain static artifacts for auditability, while the basket chart below is the interactive research surface.</p>
        </div>
      </div>
      <div class="overview-grid">
        <div class="table-card">
          <div class="table-inner">{summary_table_html}</div>
        </div>
        <div class="chart-stack">
          <div class="card">
            <div class="section-kicker">Aggregate Level</div>
            <img src="{agg_level_rel}" alt="aggregate level chart" class="aggregate-chart" />
            <p class="summary-caption">Cross-basket aggregate path built from the same output bundle used by the dashboard.</p>
          </div>
          <div class="card">
            <div class="section-kicker">Cash And Turnover</div>
            <img src="{agg_cash_rel}" alt="aggregate cash and turnover chart" class="aggregate-chart" />
            <p class="summary-caption">Reserve usage and rebalance friction at the aggregate system level.</p>
          </div>
        </div>
      </div>
    </section>

    <section class="section card">
      <div class="section-head">
        <div>
          <div class="section-kicker">Interactive Basket Paths</div>
          <h2>Selected basket profile, controls, and chart diagnostics.</h2>
          <p>Default mode shows the effective inception window and normalized tradable NAV. Full history exposes inactive prehistory. Where the real basket path is fully cash and fully unpriced, the chart may show a graph-only stochastic bridge so visual reading stays continuous without altering the underlying NAV series.</p>
        </div>
      </div>
      <div class="dashboard-shell">
        <div class="dashboard-hero-grid">
          <div class="basket-hero-panel">
            <div class="eyebrow" id="basketHeroCode">Selected Basket</div>
            <h2 id="basketHeroName">Basket profile</h2>
            <div class="hero-level" id="basketHeroLevel">—</div>
            <div class="hero-change" id="basketHeroChange"></div>
            <div class="hero-asof" id="basketHeroAsOf"></div>
            <p class="hero-copyline" id="basketHeroNarrative"></p>
            <div class="badge-row" id="basketHeroBadges"></div>
          </div>
          <aside class="profile-card">
            <div class="section-kicker">Basket Profile</div>
            <h3>Live operating frame.</h3>
            <p>These profile fields stay tied to the currently selected basket and update with the chart controls below.</p>
            <div class="profile-grid" id="basketProfileGrid"></div>
          </aside>
        </div>

        <div class="toolbar path-toolbar">
          <div class="control-block">
            <label for="basketPathSelect">Basket</label>
            <div class="basket-picker">
              <select id="basketPathSelect"></select>
              <button id="basketPrevBtn" class="pill-button" type="button">Prev Basket</button>
              <button id="basketNextBtn" class="pill-button" type="button">Next Basket</button>
            </div>
          </div>
          <div class="control-block">
            <label>History Window</label>
            <div class="segmented">
              <button type="button" data-date-mode="since_inception">Since Inception</button>
              <button type="button" data-date-mode="full_history">Full History</button>
            </div>
          </div>
          <div class="control-block">
            <label>Scale Mode</label>
            <div class="segmented">
              <button type="button" data-scale-mode="normalized">Normalized</button>
              <button type="button" data-scale-mode="raw">Raw</button>
            </div>
          </div>
          <div class="control-block">
            <label>Range</label>
            <div class="range-buttons">
              <button type="button" data-range="3M">3M</button>
              <button type="button" data-range="6M">6M</button>
              <button type="button" data-range="1Y">1Y</button>
              <button type="button" data-range="2Y">2Y</button>
              <button type="button" data-range="ALL">All</button>
            </div>
          </div>
          <div class="control-block">
            <label>Overlays</label>
            <div class="toggle-grid">
              <label class="check"><input id="toggleCash" type="checkbox" />Cash</label>
              <label class="check"><input id="toggleTurnover" type="checkbox" />Turnover</label>
              <label class="check"><input id="toggleRebalances" type="checkbox" />Rebalances</label>
              <label class="check"><input id="toggleEntriesExits" type="checkbox" />Entries / Exits</label>
            </div>
          </div>
        </div>

        <div id="basketPathMeta" class="metric-grid"></div>
        <div id="basketPathChart"></div>
      </div>
    </section>

    <section class="section card">
      <div class="section-head">
        <div>
          <div class="section-kicker">Artifacts</div>
          <h2>Canonical outputs for the current run.</h2>
          <p>These are the files this front end is reading. The website is static, so every page is inspectable against the CSV layer without a separate backend.</p>
        </div>
      </div>
      <div class="file-grid">
        <a class="file-link" href="../final_basket_list.csv">final_basket_list.csv<span>Current basket roster and headline stats</span></a>
        <a class="file-link" href="../last_year_monthly_compositions.csv">last_year_monthly_compositions.csv<span>Contract-level monthly basket compositions</span></a>
        <a class="file-link" href="../basket_level_monthly.csv">basket_level_monthly.csv<span>Daily tradable NAV, cash, coverage, and warnings</span></a>
        <a class="file-link" href="../aggregate_basket_level.csv">aggregate_basket_level.csv<span>Aggregate cross-basket index path</span></a>
        <a class="file-link" href="../basket_inception_policy.csv">basket_inception_policy.csv<span>Raw start, default inception, overrides</span></a>
        <a class="file-link" href="../monthly_cash_positions.csv">monthly_cash_positions.csv<span>Cash sleeve and rebalance context</span></a>
        <a class="file-link" href="../rebalance_transitions.csv">rebalance_transitions.csv<span>Entries, exits, and turnover deltas</span></a>
        <a class="file-link" href="../contract_lifecycle_events.csv">contract_lifecycle_events.csv<span>Lifecycle event stream and exit reasons</span></a>
      </div>
      <details>
        <summary>Open Extended Diagnostics</summary>
        <div class="file-grid" style="margin-top:12px;">
          <a class="file-link" href="../basket_monthly_summary.csv">basket_monthly_summary.csv<span>Monthly counts and concentration</span></a>
          <a class="file-link" href="../rebalance_cost_model.csv">rebalance_cost_model.csv<span>Execution-cost estimates</span></a>
          <a class="file-link" href="../factor_exposure_monthly.csv">factor_exposure_monthly.csv<span>Monthly factor overlays</span></a>
          <a class="file-link" href="../ticker_chain_registry.csv">ticker_chain_registry.csv<span>Ticker-chain membership registry</span></a>
          <a class="file-link" href="../ticker_chain_history.csv">ticker_chain_history.csv<span>Chain histories and selected months</span></a>
          <a class="file-link" href="../slot_coverage_diagnostics.csv">slot_coverage_diagnostics.csv<span>Coverage diagnostics by semantic slot</span></a>
        </div>
      </details>
    </section>
  </div>
  <script id="basket-level-data" type="application/json">{basket_level_payload_json}</script>
  <script>
    {_dashboard_chart_script()}
  </script>
</body>
</html>
"""
    (site_dir / "index.html").write_text(index_html, encoding="utf-8")

    baskets_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Itô Markets | Basket Archive</title>
  <style>
    {_site_base_css("""
    .controls {{
      display: grid;
      grid-template-columns: minmax(280px, 1.25fr) minmax(260px, 1fr) minmax(180px, 0.8fr);
      gap: 12px;
      margin-top: 14px;
    }}
    .layout {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: minmax(420px, 0.9fr) minmax(780px, 1.4fr);
      gap: 14px;
    }}
    .panel {{
      padding: 18px;
      overflow: hidden;
    }}
    .table-wrap {{
      max-height: 72vh;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #fff;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .contract-row {{
      cursor: pointer;
      transition: background 120ms ease;
    }}
    .contract-row:hover {{
      background: #f7fbff;
    }}
    .contract-row.active {{
      background: #eaf2ff;
    }}
    .contract-title {{
      font-size: 13px;
      font-weight: 700;
      color: var(--text);
      line-height: 1.35;
      margin-bottom: 4px;
    }}
    .contract-id {{
      display: inline-block;
      color: var(--muted);
      font-size: 11px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
    }}
    .weight-cell {{
      text-align: right;
      white-space: nowrap;
      min-width: 112px;
    }}
    .bar {{
      width: 100%;
      height: 6px;
      background: #e2e8f0;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 4px;
    }}
    .bar > span {{
      display: block;
      height: 100%;
      background: var(--accent);
    }}
    .main-chart {{
      width: 100%;
      margin-bottom: 12px;
    }}
    .detail-stats {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 10px;
    }}
    .stat-card {{
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
      padding: 10px 11px;
    }}
    .stat-card .label {{
      display: block;
      color: var(--muted-soft);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 5px;
      font-weight: 800;
    }}
    .stat-card .value {{
      color: var(--text);
      font-size: 17px;
      font-weight: 700;
      line-height: 1.08;
    }}
    .stat-card .sub {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.3;
    }}
    .section-title {{
      margin: 12px 0 8px;
      font-size: 11px;
      color: var(--muted-soft);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 800;
    }}
    #basketContractTrend {{
      min-height: 360px;
    }}
    .weights-grid {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #fff;
    }}
    .history-grid table {{
      table-layout: fixed;
      font-size: 11px;
    }}
    .history-grid th, .history-grid td {{
      text-align: center;
      white-space: nowrap;
      font-size: 11px;
      padding: 7px 6px;
    }}
    .history-grid th:first-child,
    .history-grid td:first-child {{
      text-align: left;
      min-width: 64px;
      font-weight: 700;
    }}
    .history-grid td.empty-month {{
      color: #94a3b8;
      background: #f8fafc;
    }}
    .history-grid td.active-month {{
      color: var(--text);
      background: #eff6ff;
      font-weight: 600;
    }}
    .chain-grid table {{
      table-layout: auto;
      font-size: 11px;
    }}
    .chain-grid th, .chain-grid td {{
      font-size: 11px;
      white-space: normal;
      vertical-align: top;
    }}
    .selected-pill {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      border: 1px solid #bfdbfe;
      background: #eff6ff;
      color: var(--accent);
      font-size: 10px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    @media (max-width: 1220px) {{
      .layout,
      .controls {{
        grid-template-columns: 1fr;
      }}
      .detail-stats {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 760px) {{
      .detail-stats {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
    """)}
  </style>
  <script src="{asset_paths['echarts_js']}"></script>
</head>
<body>
  {_site_nav("Baskets", "Contract archive and composition history")}
    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <div class="eyebrow">Basket Archive</div>
          <h1>Continuous contract history by basket.</h1>
          <p>This page is the long-form archive for every contract that has appeared in a basket. It is designed for composition analysis rather than month selection: average weight, active windows, contract detail, chain lineage, and the full monthly weight matrix sit in one place.</p>
          <div class="hero-actions">
            <span class="hero-chip"><strong>Use case</strong> Continuous archive</span>
            <span class="hero-chip"><strong>Default sort</strong> Average weight</span>
            <span class="hero-chip"><strong>Chart start</strong> First active month</span>
          </div>
        </div>
        <div class="hero-profile">
          <div class="profile-card">
            <div class="section-kicker">Reading Guide</div>
            <h3>What this page is for.</h3>
            <p>Use this view to answer which contracts dominated a basket over time, when a given contract first appeared, how large it became, and which related chain members replaced it.</p>
            <div class="profile-grid">
              <div class="profile-row"><small>Left panel</small><strong>Full contract roster in basket history</strong></div>
              <div class="profile-row"><small>Right panel</small><strong>Selected basket and contract detail</strong></div>
              <div class="profile-row"><small>Chart start</small><strong>Trimmed to first active month</strong></div>
              <div class="profile-row"><small>Chain view</small><strong>Market lineage and selected months</strong></div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="toolbar controls">
      <div>
        <label for="basketsBasketSelect">Basket</label>
        <select id="basketsBasketSelect"></select>
      </div>
      <div>
        <label for="basketsSearchInput">Search Contract</label>
        <input id="basketsSearchInput" type="text" placeholder="Search title, market id, or ticker label..." />
      </div>
      <div>
        <label for="basketsSortSelect">Sort Contracts</label>
        <select id="basketsSortSelect">
          <option value="avg_weight">Average Weight</option>
          <option value="peak_weight">Peak Weight</option>
        </select>
      </div>
    </section>

    <div class="layout">
      <section class="panel card">
        <div class="section-kicker">Contract Roster</div>
        <h2>Contracts in basket history.</h2>
        <p class="meta">The list below is not month-limited. It shows every contract that has appeared in the selected basket, sorted by weight history and searchable by market id, title, or ticker label.</p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th style="width:52px;">#</th>
                <th>Contract</th>
                <th style="width:100px;">Avg Weight</th>
                <th style="width:90px;">Starts</th>
                <th style="width:110px;">Expiry</th>
              </tr>
            </thead>
            <tbody id="basketContractsBody"></tbody>
          </table>
        </div>
      </section>

      <section class="panel card">
        <div class="section-kicker">Basket And Contract Detail</div>
        <h2>Selected basket profile.</h2>
        <div id="basketMeta" class="metric-grid"></div>
        <img id="basketCompositionChart" class="main-chart" alt="basket composition chart" />
        <div class="section-title">Monthly Basket Snapshot</div>
        <div class="weights-grid" id="basketMonthTable"></div>
        <div class="section-title">Selected Contract</div>
        <h3 id="basketDetailTitle" style="margin:0 0 6px;">Contract Detail</h3>
        <p class="meta" id="basketDetailMeta">Select a contract on the left.</p>
        <div class="detail-stats" id="basketDetailStats"></div>
        <div id="basketContractTrend"></div>
        <div class="section-title">Monthly Weight Matrix</div>
        <div class="weights-grid history-grid" id="basketWeightsGrid"></div>
        <div class="section-title">Ticker Chain History</div>
        <div class="weights-grid chain-grid" id="basketChainGrid"></div>
      </section>
    </div>
  </div>

  <script id="basket-composition-data" type="application/json">{baskets_page_payload_json}</script>
  <script>
    {_baskets_page_script()}
  </script>
</body>
</html>
"""
    (site_dir / "baskets.html").write_text(baskets_html, encoding="utf-8")

    explorer_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Itô Markets | Month Explorer</title>
  <style>
    {_site_base_css("""
    .controls {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: minmax(280px, 1.35fr) minmax(160px, 0.75fr) minmax(280px, 1.2fr) repeat(3, minmax(135px, 0.8fr));
      gap: 12px;
      align-items: end;
    }}
    .layout {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: minmax(420px, 0.9fr) minmax(760px, 1.4fr);
      gap: 14px;
    }}
    .panel {{
      padding: 18px;
      overflow: hidden;
    }}
    .table-wrap {{
      max-height: 74vh;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #fff;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .contract-row {{
      cursor: pointer;
      transition: background 120ms ease;
    }}
    .contract-row:hover {{
      background: #f7fbff;
    }}
    .contract-row.active {{
      background: #eaf2ff;
    }}
    .contract-title {{
      font-size: 13px;
      font-weight: 700;
      color: var(--text);
      line-height: 1.35;
      margin-bottom: 4px;
    }}
    .contract-id {{
      display: inline-block;
      color: var(--muted);
      font-size: 11px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
    }}
    .weight-cell {{
      text-align: right;
      white-space: nowrap;
      min-width: 112px;
    }}
    .bar {{
      width: 100%;
      height: 6px;
      background: #e2e8f0;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 4px;
    }}
    .bar > span {{
      display: block;
      height: 100%;
      background: var(--accent);
    }}
    .detail-stats {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 10px;
    }}
    .stat-card {{
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
      padding: 10px 11px;
    }}
    .stat-card .label {{
      display: block;
      color: var(--muted-soft);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 5px;
      font-weight: 800;
    }}
    .stat-card .value {{
      color: var(--text);
      font-size: 17px;
      font-weight: 700;
      line-height: 1.08;
    }}
    .stat-card .sub {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.3;
    }}
    .section-title {{
      margin: 12px 0 8px;
      font-size: 11px;
      color: var(--muted-soft);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 800;
    }}
    #contractTrend {{
      min-height: 360px;
    }}
    .weights-grid {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #fff;
    }}
    .history-grid table {{
      table-layout: fixed;
      font-size: 11px;
    }}
    .history-grid th, .history-grid td {{
      text-align: center;
      white-space: nowrap;
      font-size: 11px;
      padding: 7px 6px;
    }}
    .history-grid th:first-child,
    .history-grid td:first-child {{
      text-align: left;
      min-width: 64px;
      font-weight: 700;
    }}
    .history-grid td.empty-month {{
      color: #94a3b8;
      background: #f8fafc;
    }}
    .history-grid td.active-month {{
      color: var(--text);
      background: #eff6ff;
      font-weight: 600;
    }}
    .history-grid td.selected-month {{
      background: #dbeafe;
      box-shadow: inset 0 0 0 1px #60a5fa;
    }}
    .chain-grid table {{
      table-layout: auto;
      font-size: 11px;
    }}
    .chain-grid th, .chain-grid td {{
      font-size: 11px;
      white-space: normal;
      vertical-align: top;
    }}
    .selected-pill {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      border: 1px solid #bfdbfe;
      background: #eff6ff;
      color: var(--accent);
      font-size: 10px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    @media (max-width: 1220px) {{
      .layout,
      .controls {{
        grid-template-columns: 1fr;
      }}
      .table-wrap {{
        max-height: 44vh;
      }}
      .detail-stats {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 760px) {{
      .detail-stats {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
    """)}
  </style>
  <script src="{asset_paths['echarts_js']}"></script>
</head>
<body>
  {_site_nav("Explorer", "Month-by-month contract analysis")}
    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <div class="eyebrow">Month Explorer</div>
          <h1>Inspect exactly which contracts were live in each basket month.</h1>
          <p>This page is built for month-specific audit work. Pick a basket and month, scan the selected contract slate, and drill into weight history, direction, chain lineage, and the exact contract metadata used for that point in time.</p>
          <div class="hero-actions">
            <span class="hero-chip"><strong>Use case</strong> Single-month audit</span>
            <span class="hero-chip"><strong>Input</strong> Generated basket-month compositions</span>
            <span class="hero-chip"><strong>Detail</strong> Direction + chain history</span>
          </div>
        </div>
        <div class="hero-profile">
          <div class="profile-card">
            <div class="section-kicker">Reading Guide</div>
            <h3>How to use this screen.</h3>
            <p>Start with the left table for the chosen month. Then use the right panel to understand the selected contract in context: historical weight path, chain replacements, and the exact direction semantics carried into the basket.</p>
            <div class="profile-grid">
              <div class="profile-row"><small>Month</small><strong>Exact rebalance slice</strong></div>
              <div class="profile-row"><small>Contract list</small><strong>Only contracts selected that month</strong></div>
              <div class="profile-row"><small>Trend chart</small><strong>Weight history for selected contract</strong></div>
              <div class="profile-row"><small>Chain history</small><strong>Lineage of related expiry markets</strong></div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="toolbar controls">
      <div>
        <label for="basketSelect">Basket</label>
        <select id="basketSelect"></select>
      </div>
      <div>
        <label for="monthSelect">Month</label>
        <select id="monthSelect"></select>
      </div>
      <div>
        <label for="searchInput">Search Contract</label>
        <input id="searchInput" type="text" placeholder="Search market id or contract text..." />
      </div>
      <div>
        <label>&nbsp;</label>
        <button id="prevMonthBtn" type="button">Prev Month</button>
      </div>
      <div>
        <label>&nbsp;</label>
        <button id="nextMonthBtn" type="button">Next Month</button>
      </div>
      <div>
        <label>&nbsp;</label>
        <button id="clearSearchBtn" type="button">Clear Search</button>
      </div>
    </section>

    <div class="layout">
      <section class="panel card">
        <div class="section-kicker">Selected Month</div>
        <h2>Contracts in selected month.</h2>
        <p class="meta" id="leftMeta"></p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th style="width:52px;">#</th>
                <th>Contract</th>
                <th style="width:95px;">Weight</th>
                <th style="width:110px;">Expiry</th>
              </tr>
            </thead>
            <tbody id="contractsBody"></tbody>
          </table>
        </div>
      </section>

      <section class="panel card">
        <div class="section-kicker">Selected Contract</div>
        <h2 id="detailTitle">Contract Detail</h2>
        <p class="meta" id="detailMeta">Select a contract on the left.</p>
        <div class="detail-stats" id="detailStats"></div>
        <div class="section-title">Weight History</div>
        <div id="contractTrend"></div>
        <div class="section-title">Monthly Weight Matrix</div>
        <div class="weights-grid history-grid" id="weightsGrid"></div>
        <div class="section-title">Ticker Chain History</div>
        <div class="weights-grid chain-grid" id="chainGrid"></div>
      </section>
    </div>
  </div>

  <script id="explorer-data" type="application/json">{explorer_payload_json}</script>
  <script>
    {_explorer_chart_script()}
  </script>
</body>
</html>
"""
    (site_dir / "explorer.html").write_text(explorer_html, encoding="utf-8")

    _build_methodology_html(
        output_path=site_dir / "methodology.html",
        specs=specs,
        summary=summary,
        compositions=compositions,
        start=start,
        end=end,
        monthly_summary=monthly_summary,
        cash_positions=cash_positions,
        transitions=transitions,
        factor_exposure=factor_exposure,
        ticker_registry=ticker_chain_meta,
        ticker_history=ticker_chain_history,
        lifecycle_events=lifecycle_events,
        cost_model=cost_model,
        aggregate_series=agg,
        aggregate_level_chart_rel=agg_level_rel if not agg.empty else None,
        aggregate_cash_chart_rel=agg_cash_rel if not agg.empty else None,
        inception_policy=inception_df,
    )


def _build_ticker_chain_outputs(
    compositions: pd.DataFrame,
    chain_meta: pd.DataFrame,
    chain_markets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if compositions.empty:
        return pd.DataFrame(), pd.DataFrame()

    c = compositions.copy()
    c["rebalance_date"] = pd.to_datetime(c["rebalance_date"], errors="coerce")
    c["end_date"] = pd.to_datetime(c["end_date"], errors="coerce")
    non_cash = c[~c["is_cash"]].copy()
    if non_cash.empty:
        return pd.DataFrame(), pd.DataFrame()

    reg = (
        non_cash.groupby(
            [
                "domain",
                "basket_code",
                "basket_name",
                "ticker_id",
                "market_id",
                "title",
                "event_slug",
                "event_family_key",
                "llm_category",
            ],
            as_index=False,
        )
        .agg(
            selected_first_rebalance=("rebalance_date", "min"),
            selected_last_rebalance=("rebalance_date", "max"),
            selected_months=("rebalance_date", "nunique"),
            avg_target_weight=("target_weight", "mean"),
            max_target_weight=("target_weight", "max"),
            avg_days_to_expiry=("days_to_expiry", "mean"),
        )
    )
    reg["selected_first_rebalance"] = pd.to_datetime(reg["selected_first_rebalance"]).dt.date.astype(str)
    reg["selected_last_rebalance"] = pd.to_datetime(reg["selected_last_rebalance"]).dt.date.astype(str)

    if chain_meta is not None and not chain_meta.empty:
        reg = reg.merge(chain_meta, on="ticker_id", how="left")
    else:
        reg["ticker_name"] = np.nan
        reg["chain_market_count"] = np.nan
        reg["chain_first_end_date"] = np.nan
        reg["chain_last_end_date"] = np.nan
        reg["chain_event_slugs"] = np.nan

    history = pd.DataFrame()
    if chain_markets is not None and not chain_markets.empty:
        selected_tickers = set(reg["ticker_id"].astype(str))
        selected_markets = set(non_cash["market_id"].astype(str))
        hist = chain_markets[chain_markets["ticker_id"].astype(str).isin(selected_tickers)].copy()
        if not hist.empty:
            hist["end_date"] = pd.to_datetime(hist["end_date"], errors="coerce")
            hist["chain_order"] = (
                hist.sort_values(["ticker_id", "end_date", "market_id"])
                .groupby("ticker_id")
                .cumcount()
                + 1
            )
            selected_dates = (
                non_cash.groupby("market_id")["rebalance_date"]
                .apply(lambda s: ", ".join(sorted({pd.Timestamp(x).date().isoformat() for x in s.dropna()})))
                .to_dict()
            )
            hist["selected_any"] = hist["market_id"].astype(str).isin(selected_markets)
            hist["selected_rebalance_dates"] = hist["market_id"].astype(str).map(selected_dates).fillna("")
            hist["end_date"] = hist["end_date"].dt.date.astype(str).replace("NaT", "")
            history = hist.sort_values(["ticker_id", "chain_order", "market_id"]).reset_index(drop=True)

    reg = reg.sort_values(["domain", "basket_code", "ticker_id", "market_id"]).reset_index(drop=True)
    return reg, history


def _build_lifecycle_events(compositions: pd.DataFrame) -> pd.DataFrame:
    if compositions.empty:
        return pd.DataFrame()

    c = compositions.copy()
    c["rebalance_date"] = pd.to_datetime(c["rebalance_date"], errors="coerce")
    c["end_date"] = pd.to_datetime(c["end_date"], errors="coerce")
    if "current_price" in c.columns:
        c["current_price"] = pd.to_numeric(c["current_price"], errors="coerce")
    else:
        c["current_price"] = np.nan
    c["effective_price"] = pd.to_numeric(c.get("effective_price", np.nan), errors="coerce")

    non_cash = c[~c["is_cash"]].copy()
    if non_cash.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for code, grp in non_cash.groupby("basket_code", sort=True):
        grp = grp.sort_values(["rebalance_date", "target_weight"], ascending=[True, False]).copy()
        prev_date: pd.Timestamp | None = None
        prev_map: pd.DataFrame | None = None
        for d, dg in grp.groupby("rebalance_date", sort=True):
            cur = dg.set_index("market_id")
            if prev_map is not None and prev_date is not None:
                prev_ids = set(prev_map.index)
                cur_ids = set(cur.index)
                exited = sorted(prev_ids - cur_ids)
                entered = sorted(cur_ids - prev_ids)

                for mid in exited:
                    r = prev_map.loc[mid]
                    end_dt = pd.to_datetime(r.get("end_date"), errors="coerce")
                    px = pd.to_numeric(r.get("effective_price"), errors="coerce")
                    if pd.isna(px):
                        px = pd.to_numeric(r.get("current_price"), errors="coerce")
                    is_expired = bool(pd.notna(end_dt) and pd.Timestamp(d) > pd.Timestamp(end_dt))
                    is_certainty = bool(pd.notna(px) and (float(px) <= 0.05 or float(px) >= 0.95))
                    reason = "expiry_or_resolution" if is_expired else ("certainty_threshold" if is_certainty else "rebalance_replace")
                    rows.append(
                        {
                            "rebalance_date": pd.Timestamp(d).date().isoformat(),
                            "basket_code": code,
                            "domain": str(r.get("domain", "")),
                            "basket_name": str(r.get("basket_name", "")),
                            "action": "EXIT",
                            "market_id": str(mid),
                            "ticker_id": str(r.get("ticker_id", "")),
                            "title": str(r.get("title", "")),
                            "event_slug": str(r.get("event_slug", "")),
                            "position_side": str(r.get("position_side", "YES")),
                            "end_date": end_dt.date().isoformat() if pd.notna(end_dt) else "",
                            "days_to_expiry_at_prev": float(r.get("days_to_expiry", np.nan)),
                            "price_at_prev": float(px) if pd.notna(px) else np.nan,
                            "reason": reason,
                            "from_rebalance_date": pd.Timestamp(prev_date).date().isoformat(),
                        }
                    )

                for mid in entered:
                    r = cur.loc[mid]
                    rows.append(
                        {
                            "rebalance_date": pd.Timestamp(d).date().isoformat(),
                            "basket_code": code,
                            "domain": str(r.get("domain", "")),
                            "basket_name": str(r.get("basket_name", "")),
                            "action": "ENTRY",
                            "market_id": str(mid),
                            "ticker_id": str(r.get("ticker_id", "")),
                            "title": str(r.get("title", "")),
                            "event_slug": str(r.get("event_slug", "")),
                            "position_side": str(r.get("position_side", "YES")),
                            "end_date": pd.Timestamp(r.get("end_date")).date().isoformat() if pd.notna(r.get("end_date")) else "",
                            "days_to_expiry_at_prev": np.nan,
                            "price_at_prev": np.nan,
                            "reason": "new_selection",
                            "from_rebalance_date": pd.Timestamp(prev_date).date().isoformat(),
                        }
                    )

            prev_map = cur
            prev_date = pd.Timestamp(d)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["rebalance_date", "basket_code", "action", "market_id"]).reset_index(drop=True)


def _build_rebalance_cost_model(compositions: pd.DataFrame) -> pd.DataFrame:
    if compositions.empty:
        return pd.DataFrame()

    c = compositions.copy()
    c["rebalance_date"] = pd.to_datetime(c["rebalance_date"], errors="coerce")
    c["days_to_expiry"] = pd.to_numeric(c["days_to_expiry"], errors="coerce")
    if "current_price" in c.columns:
        c["current_price"] = pd.to_numeric(c["current_price"], errors="coerce")
    else:
        c["current_price"] = np.nan
    c["effective_price"] = pd.to_numeric(c.get("effective_price", np.nan), errors="coerce")

    non_cash = c[~c["is_cash"]].copy()
    cash = c[c["is_cash"]][["rebalance_date", "basket_code", "target_weight", "turnover"]].rename(
        columns={"target_weight": "cash_weight"}
    )
    if non_cash.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for keys, g in non_cash.groupby(["rebalance_date", "domain", "basket_code", "basket_name"], sort=True):
        d, domain, code, name = keys
        turnover = float(g["turnover"].iloc[0]) if "turnover" in g.columns and len(g) else 0.0
        avg_dte = float(g["days_to_expiry"].mean()) if g["days_to_expiry"].notna().any() else np.nan
        expiring_30d_share = float((g["days_to_expiry"] <= 30).mean()) if g["days_to_expiry"].notna().any() else 0.0
        expiring_60d_share = float((g["days_to_expiry"] <= 60).mean()) if g["days_to_expiry"].notna().any() else 0.0
        price_for_certainty = g["effective_price"] if g["effective_price"].notna().any() else g["current_price"]
        certainty_share = float(((price_for_certainty <= 0.05) | (price_for_certainty >= 0.95)).mean()) if price_for_certainty.notna().any() else np.nan

        base_bps = 18.0
        turnover_bps = 35.0 * turnover
        expiry_bps = 10.0 * expiring_30d_share + 6.0 * expiring_60d_share
        certainty_bps = 25.0 * certainty_share if pd.notna(certainty_share) else 0.0
        est_bps = base_bps + turnover_bps + expiry_bps + certainty_bps
        est_drag = turnover * est_bps / 10000.0
        rows.append(
            {
                "rebalance_date": pd.Timestamp(d).date().isoformat(),
                "domain": domain,
                "basket_code": code,
                "basket_name": name,
                "avg_days_to_expiry": avg_dte,
                "expiring_30d_share": expiring_30d_share,
                "expiring_60d_share": expiring_60d_share,
                "certainty_share": certainty_share,
                "turnover": turnover,
                "base_cost_bps": base_bps,
                "turnover_component_bps": turnover_bps,
                "expiry_component_bps": expiry_bps,
                "certainty_component_bps": certainty_bps,
                "estimated_total_cost_bps": est_bps,
                "estimated_cost_drag_return": est_drag,
            }
        )

    out = pd.DataFrame(rows)
    if not cash.empty and not out.empty:
        cash["rebalance_date"] = pd.to_datetime(cash["rebalance_date"]).dt.date.astype(str)
        out = out.merge(
            cash[["rebalance_date", "basket_code", "cash_weight"]].drop_duplicates(["rebalance_date", "basket_code"]),
            on=["rebalance_date", "basket_code"],
            how="left",
        )
    return out.sort_values(["rebalance_date", "domain", "basket_code"]).reset_index(drop=True)


def _prune_unused_outputs(output_dir: Path) -> None:
    out = output_dir.resolve()
    if out.name != "prediction_basket":
        raise ValueError(f"Refusing prune on unexpected path: {out}")

    keep_files = {
        "final_basket_list.csv",
        "last_year_monthly_compositions.csv",
        "basket_monthly_summary.csv",
        "basket_level_monthly.csv",
        "basket_inception_policy.csv",
        "slot_coverage_diagnostics.csv",
        "monthly_cash_positions.csv",
        "aggregate_basket_level.csv",
        "rebalance_cost_model.csv",
        "rebalance_transitions.csv",
        "contract_lifecycle_events.csv",
        "factor_exposure_monthly.csv",
        "ticker_chain_registry.csv",
        "ticker_chain_history.csv",
    }
    keep_dirs = {"website"}

    for child in out.iterdir():
        if child.name in keep_files or child.name in keep_dirs:
            continue
        _safe_delete(child)


def _refresh_exposure_direction_cache(
    processed_dir: str | Path,
    *,
    force: bool = False,
    batch_size: int = 30,
    max_workers: int = 8,
    model: str | None = None,
) -> None:
    from src.exposure.side_detection import classify_all_markets

    processed = Path(processed_dir)
    mapping_path = processed / "ticker_mapping.parquet"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing market universe for exposure refresh: {mapping_path}")
    universe = pd.read_parquet(mapping_path)
    if "market_id" not in universe.columns or "title" not in universe.columns:
        raise ValueError(f"{mapping_path} must contain market_id and title columns")
    classify_all_markets(
        universe[["market_id", "title"]].drop_duplicates("market_id"),
        title_col="title",
        id_col="market_id",
        batch_size=batch_size,
        max_workers=max_workers,
        force_reclassify=force,
        model=model,
    )


def run_thematic_generation(
    processed_dir: str | Path = "data/processed",
    output_dir: str | Path = "data/outputs/prediction_basket",
    start: str = "2025-03-01",
    end: str = "2026-02-01",
    build_site: bool = True,
    prune_unused: bool = False,
    strict_temporal: bool = True,
    require_temporal_history: bool = True,
    refresh_exposure_directions: bool = False,
    force_exposure_refresh: bool = False,
    exposure_batch_size: int = 30,
    exposure_max_workers: int = 8,
    exposure_model: str | None = None,
    log_progress: bool = False,
) -> dict[str, str]:
    if refresh_exposure_directions:
        _refresh_exposure_direction_cache(
            processed_dir,
            force=force_exposure_refresh,
            batch_size=exposure_batch_size,
            max_workers=exposure_max_workers,
            model=exposure_model,
        )
    universe = load_market_universe(
        processed_dir,
        require_temporal_history=require_temporal_history,
    )
    chain_meta, chain_markets = load_ticker_chains(processed_dir)
    builder = ThematicBasketBuilder(
        universe=universe,
        specs=default_specs(),
        strict_temporal=strict_temporal,
    )

    rebalance_dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS")
    compositions = builder.build_for_dates(rebalance_dates, log_progress=log_progress)

    root_out = Path(output_dir)
    out_dir = root_out
    root_out.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    compositions = attach_asof_prices(compositions, processed_dir)

    compositions_csv = out_dir / "last_year_monthly_compositions.csv"
    compositions.to_csv(compositions_csv, index=False)

    summary = summarize_latest_baskets(compositions)
    summary_csv = out_dir / "final_basket_list.csv"
    summary.to_csv(summary_csv, index=False)

    monthly_summary_path = out_dir / "basket_monthly_summary.csv"
    basket_level_path = out_dir / "basket_level_monthly.csv"
    cash_positions_path = out_dir / "monthly_cash_positions.csv"
    transitions_path = out_dir / "rebalance_transitions.csv"
    factor_exposure_path = out_dir / "factor_exposure_monthly.csv"
    ticker_registry_path = out_dir / "ticker_chain_registry.csv"
    ticker_history_path = out_dir / "ticker_chain_history.csv"
    lifecycle_path = out_dir / "contract_lifecycle_events.csv"
    cost_model_path = out_dir / "rebalance_cost_model.csv"
    aggregate_path = out_dir / "aggregate_basket_level.csv"
    inception_policy_path = out_dir / "basket_inception_policy.csv"
    slot_diag_path = out_dir / "slot_coverage_diagnostics.csv"

    if not compositions.empty:
        c = compositions.copy()
        c["rebalance_date"] = pd.to_datetime(c["rebalance_date"])
        non_cash = c[~c["is_cash"]].copy()
        cash = c[c["is_cash"]][
            ["rebalance_date", "basket_code", "target_weight", "turnover", "treasury_risk", "broad_risk"]
        ].rename(
            columns={
                "target_weight": "cash_weight",
            }
        )
        top_contracts = (
            non_cash.sort_values(["rebalance_date", "basket_code", "target_weight"], ascending=[True, True, False])
            .groupby(["rebalance_date", "basket_code"])
            .head(5)
            .groupby(["rebalance_date", "basket_code"])["market_id"]
            .apply(lambda s: ", ".join(s.astype(str)))
            .rename("top_5_contract_ids")
            .reset_index()
        )
        monthly_summary = (
            non_cash.groupby(
                ["rebalance_date", "domain", "basket_code", "basket_name", "basket_weight"], as_index=False
            )
            .agg(
                n_contracts=("market_id", "nunique"),
                avg_contract_weight=("target_weight", "mean"),
                max_contract_weight=("target_weight", "max"),
            )
            .merge(cash, on=["rebalance_date", "basket_code"], how="left")
            .merge(top_contracts, on=["rebalance_date", "basket_code"], how="left")
            .sort_values(["rebalance_date", "domain", "basket_code"])
        )
        monthly_summary.to_csv(monthly_summary_path, index=False)

        cash_positions = monthly_summary[
            [
                "rebalance_date",
                "domain",
                "basket_code",
                "basket_name",
                "cash_weight",
                "turnover",
                "treasury_risk",
                "broad_risk",
                "n_contracts",
            ]
        ].copy()
        cash_positions.to_csv(cash_positions_path, index=False)

        # Rebalance transition report (entries/exits/upsizes/downsizes each month).
        trans_rows: list[dict] = []
        for code, grp in non_cash.groupby("basket_code", sort=True):
            prev_w = pd.Series(dtype=float)
            prev_set = set()
            for d, dg in grp.groupby("rebalance_date", sort=True):
                cur = dg.set_index("market_id")["target_weight"].astype(float)
                cur_set = set(cur.index)
                entered = sorted(cur_set - prev_set)
                exited = sorted(prev_set - cur_set)
                overlap = sorted(cur_set & prev_set)
                up = [m for m in overlap if float(cur[m] - prev_w[m]) > 1e-9]
                down = [m for m in overlap if float(cur[m] - prev_w[m]) < -1e-9]
                union = sorted(cur_set | prev_set)
                turnover = float((cur.reindex(union, fill_value=0.0) - prev_w.reindex(union, fill_value=0.0)).abs().sum() / 2.0)
                trans_rows.append(
                    {
                        "rebalance_date": pd.Timestamp(d).date().isoformat(),
                        "basket_code": code,
                        "n_current": int(len(cur_set)),
                        "turnover": turnover,
                        "entries_n": len(entered),
                        "exits_n": len(exited),
                        "upsizes_n": len(up),
                        "downsizes_n": len(down),
                        "entries": ", ".join(entered),
                        "exits": ", ".join(exited),
                        "upsizes": ", ".join(up[:12]),
                        "downsizes": ", ".join(down[:12]),
                    }
                )
                prev_w = cur
                prev_set = cur_set
        transitions_df = pd.DataFrame(trans_rows)
        transitions_df.to_csv(transitions_path, index=False)
        basket_level_df, aggregate_df, spot_diag_df = build_daily_basket_nav(
            c,
            processed_dir,
            return_spot_diagnostics=True,
            log_progress=log_progress,
        )
        if basket_level_df.empty:
            basket_level_df = _build_basket_level_series(c, transitions_df)
            aggregate_df = _build_aggregate_basket_series(c, transitions_df)
            spot_diag_df = pd.DataFrame()
        else:
            basket_level_df["rebalance_date"] = pd.to_datetime(basket_level_df["rebalance_date"], errors="coerce")
            aggregate_df["rebalance_date"] = pd.to_datetime(aggregate_df["rebalance_date"], errors="coerce")
        basket_level_df = basket_level_df.drop(
            columns=[
                "spot_risk_level",
                "spot_weighted_effective_price_raw",
                "spot_weighted_effective_price_horizon",
                "spot_weighted_dte_days",
                "spot_price_coverage_share",
                "spot_stale_weight_share",
                "nav_spot_gap",
                "tenor_target_days",
                "tenor_drift_days",
            ],
            errors="ignore",
        )
        aggregate_df = aggregate_df.drop(
            columns=[
                "overall_spot_risk_level",
                "overall_spot_price_coverage_share",
                "overall_nav_spot_gap",
            ],
            errors="ignore",
        )
        basket_level_df.to_csv(basket_level_path, index=False)
        aggregate_df.to_csv(aggregate_path, index=False)
        inception_policy_df = _build_inception_policy(basket_level_df, builder.specs, _load_inception_overrides(builder.specs))
        inception_policy_df.to_csv(inception_policy_path, index=False)
        slot_diag_df = builder.last_slot_coverage_diagnostics.copy() if isinstance(builder.last_slot_coverage_diagnostics, pd.DataFrame) else pd.DataFrame()
        slot_diag_df.to_csv(slot_diag_path, index=False)

        # Monthly factor exposures from selected non-cash contracts.
        beta_cols = [col for col in non_cash.columns if col.startswith("beta_")]
        if beta_cols:
            exp_df = non_cash.copy()
            for col in beta_cols:
                exp_df[col] = pd.to_numeric(exp_df[col], errors="coerce").fillna(0.0)
            grouped = exp_df.groupby(["rebalance_date", "domain", "basket_code", "basket_name"], as_index=False)
            rows = []
            for keys, g in grouped:
                out = {
                    "rebalance_date": pd.Timestamp(keys[0]).date().isoformat(),
                    "domain": keys[1],
                    "basket_code": keys[2],
                    "basket_name": keys[3],
                }
                for col in beta_cols:
                    out[col] = float((g[col] * g["target_weight"]).sum())
                rows.append(out)
            factor_exp = pd.DataFrame(rows).sort_values(["rebalance_date", "domain", "basket_code"])
            # explicit treasury composite risk proxy
            tre_cols = [c for c in beta_cols if c in {"beta_TNX", "beta_TLT", "beta_SHY", "beta_TLH", "beta_TYX", "beta_FVX", "beta_IRX"}]
            if tre_cols:
                factor_exp["treasury_exposure_abs"] = factor_exp[tre_cols].abs().sum(axis=1)
            factor_exp.to_csv(factor_exposure_path, index=False)
        else:
            pd.DataFrame().to_csv(factor_exposure_path, index=False)

        # Ticker chain outputs and lifecycle/cost diagnostics.
        ticker_registry, ticker_history = _build_ticker_chain_outputs(c, chain_meta, chain_markets)
        ticker_registry.to_csv(ticker_registry_path, index=False)
        ticker_history.to_csv(ticker_history_path, index=False)

        lifecycle_df = _build_lifecycle_events(c)
        lifecycle_df.to_csv(lifecycle_path, index=False)

        cost_df = _build_rebalance_cost_model(c)
        cost_df.to_csv(cost_model_path, index=False)
    else:
        pd.DataFrame().to_csv(monthly_summary_path, index=False)
        pd.DataFrame().to_csv(basket_level_path, index=False)
        pd.DataFrame().to_csv(cash_positions_path, index=False)
        pd.DataFrame().to_csv(aggregate_path, index=False)
        pd.DataFrame().to_csv(transitions_path, index=False)
        pd.DataFrame().to_csv(factor_exposure_path, index=False)
        pd.DataFrame().to_csv(ticker_registry_path, index=False)
        pd.DataFrame().to_csv(ticker_history_path, index=False)
        pd.DataFrame().to_csv(lifecycle_path, index=False)
        pd.DataFrame().to_csv(cost_model_path, index=False)
        pd.DataFrame().to_csv(slot_diag_path, index=False)
        _build_inception_policy(pd.DataFrame(), builder.specs, _load_inception_overrides(builder.specs)).to_csv(inception_policy_path, index=False)

    if build_site:
        monthly_summary = pd.read_csv(root_out / "basket_monthly_summary.csv")
        basket_level_df = pd.read_csv(root_out / "basket_level_monthly.csv") if (root_out / "basket_level_monthly.csv").exists() else pd.DataFrame()
        cash_positions_df = pd.read_csv(root_out / "monthly_cash_positions.csv") if (root_out / "monthly_cash_positions.csv").exists() else pd.DataFrame()
        aggregate_df = pd.read_csv(root_out / "aggregate_basket_level.csv") if (root_out / "aggregate_basket_level.csv").exists() else pd.DataFrame()
        transitions_df = pd.read_csv(root_out / "rebalance_transitions.csv") if (root_out / "rebalance_transitions.csv").exists() else pd.DataFrame()
        factor_exposure_df = pd.read_csv(root_out / "factor_exposure_monthly.csv") if (root_out / "factor_exposure_monthly.csv").exists() else pd.DataFrame()
        ticker_registry_df = pd.read_csv(root_out / "ticker_chain_registry.csv") if (root_out / "ticker_chain_registry.csv").exists() else pd.DataFrame()
        ticker_history_df = pd.read_csv(root_out / "ticker_chain_history.csv") if (root_out / "ticker_chain_history.csv").exists() else pd.DataFrame()
        lifecycle_df = pd.read_csv(root_out / "contract_lifecycle_events.csv") if (root_out / "contract_lifecycle_events.csv").exists() else pd.DataFrame()
        cost_model_df = pd.read_csv(root_out / "rebalance_cost_model.csv") if (root_out / "rebalance_cost_model.csv").exists() else pd.DataFrame()
        inception_policy_df = pd.read_csv(root_out / "basket_inception_policy.csv") if (root_out / "basket_inception_policy.csv").exists() else pd.DataFrame()
        _build_website_html(
            site_dir=root_out / "website",
            summary=summary,
            monthly_summary=monthly_summary,
            compositions=compositions,
            specs=builder.specs,
            start=start,
            end=end,
            ticker_chain_meta=ticker_registry_df,
            ticker_chain_history=ticker_history_df,
            basket_level_series=basket_level_df,
            aggregate_series=aggregate_df,
            cash_positions=cash_positions_df,
            transitions=transitions_df,
            factor_exposure=factor_exposure_df,
            lifecycle_events=lifecycle_df,
            cost_model=cost_model_df,
            inception_policy=inception_policy_df,
        )

    if prune_unused:
        _prune_unused_outputs(root_out)

    return {
        "compositions_csv": str((root_out / "last_year_monthly_compositions.csv").resolve()),
        "summary_csv": str((root_out / "final_basket_list.csv").resolve()),
        "basket_level_monthly_csv": str((root_out / "basket_level_monthly.csv").resolve()),
        "cash_positions_csv": str((root_out / "monthly_cash_positions.csv").resolve()),
        "aggregate_basket_level_csv": str((root_out / "aggregate_basket_level.csv").resolve()),
        "rebalance_cost_model_csv": str((root_out / "rebalance_cost_model.csv").resolve()),
        "basket_inception_policy_csv": str((root_out / "basket_inception_policy.csv").resolve()),
        "rebalance_transitions_csv": str((root_out / "rebalance_transitions.csv").resolve()),
        "contract_lifecycle_events_csv": str((root_out / "contract_lifecycle_events.csv").resolve()),
        "factor_exposure_csv": str((root_out / "factor_exposure_monthly.csv").resolve()),
        "ticker_chain_registry_csv": str((root_out / "ticker_chain_registry.csv").resolve()),
        "ticker_chain_history_csv": str((root_out / "ticker_chain_history.csv").resolve()),
        "website_index": str((root_out / "website" / "index.html").resolve()),
        "website_explorer": str((root_out / "website" / "explorer.html").resolve()),
        "website_baskets": str((root_out / "website" / "baskets.html").resolve()),
        "website_methodology": str((root_out / "website" / "methodology.html").resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thematic monthly prediction-market baskets")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/outputs/prediction_basket")
    parser.add_argument("--start", default="2025-03-01")
    parser.add_argument("--end", default="2026-02-01")
    parser.add_argument("--no-site", action="store_true", help="Skip HTML site generation")
    parser.add_argument("--prune-unused", action="store_true", help="Delete non-semantic legacy outputs")
    parser.add_argument(
        "--allow-missing-temporal-history",
        action="store_true",
        help="Allow generation when listing-date coverage is incomplete (not recommended)",
    )
    parser.add_argument(
        "--no-strict-temporal",
        action="store_true",
        help="Disable strict as-of listing filter (not recommended)",
    )
    parser.add_argument(
        "--refresh-exposure-directions",
        action="store_true",
        help="Refresh the OpenAI direction cache before basket generation",
    )
    parser.add_argument(
        "--force-exposure-refresh",
        action="store_true",
        help="Ignore cached direction classifications and refetch them",
    )
    parser.add_argument(
        "--exposure-batch-size",
        type=int,
        default=30,
        help="Markets per OpenAI direction request when refreshing the cache",
    )
    parser.add_argument(
        "--exposure-max-workers",
        type=int,
        default=8,
        help="Concurrent OpenAI direction requests when refreshing the cache",
    )
    parser.add_argument(
        "--exposure-model",
        default=None,
        help="Override OpenAI model for direction classification (defaults to OPENAI_EXPOSURE_MODEL or gpt-4.1-mini)",
    )
    parser.add_argument(
        "--log-progress",
        action="store_true",
        help="Print percentage progress while rebuilding historical basket selections",
    )
    args = parser.parse_args()

    outputs = run_thematic_generation(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        build_site=not args.no_site,
        prune_unused=args.prune_unused,
        strict_temporal=not args.no_strict_temporal,
        require_temporal_history=not args.allow_missing_temporal_history,
        refresh_exposure_directions=args.refresh_exposure_directions,
        force_exposure_refresh=args.force_exposure_refresh,
        exposure_batch_size=args.exposure_batch_size,
        exposure_max_workers=args.exposure_max_workers,
        exposure_model=args.exposure_model,
        log_progress=args.log_progress,
    )
    print("Thematic basket generation complete.")
    for k, v in outputs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
