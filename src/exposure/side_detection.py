"""LLM-based exposure/side detection for prediction market titles.

Uses the OpenAI API to determine what buying the YES token means in real-world
terms and, importantly for thematic baskets, whether a YES resolution is
``risk_up`` / ``risk_down`` or ``growth_up`` / ``growth_down``.

The cache is backward compatible with older entries that only stored
``exposure_direction``.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import openai

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "exposure_classifications.json"
DEFAULT_MODEL = os.getenv("OPENAI_EXPOSURE_MODEL", "gpt-4.1-mini")
VALID_OUTCOME_POLARITIES = {
    "risk_up",
    "risk_down",
    "growth_up",
    "growth_down",
    "neutral",
    "ambiguous",
}

SYSTEM_PROMPT = """You are an expert at analyzing prediction market titles to determine the real-world exposure of buying the YES token.

For each market, determine:
1. exposure_direction: "long" or "short"
   - "long" = buying YES means you profit when the described event/outcome HAPPENS or increases
   - "short" = buying YES means you profit when something NEGATIVE happens (recession, decline, failure, loss, etc.)

2. yes_outcome_polarity:
   - "risk_up" = YES means higher geopolitical / economic / disruption risk
   - "risk_down" = YES means de-escalation / stabilization / normalization
   - "growth_up" = YES means stronger growth / capability / adoption / upside
   - "growth_down" = YES means slower growth / recession / weakness / downside
   - "neutral" = YES is not clearly risk or growth directional
   - "ambiguous" = title is too unclear to classify confidently

3. exposure_description: A one-line description of what buying YES means

4. yes_outcome_reason: A short reason explaining the yes_outcome_polarity

5. confidence: 0.0-1.0 how confident you are

Key rules:
- "Will X happen?" → usually "long" (betting X happens)
- "Will X NOT happen?" or "Will X fail?" → depends on what X is
- Double negatives: "Will Bitcoin NOT fall below 50K?" → "long" (betting on Bitcoin strength)
- Negative outcomes: "Will unemployment rise?" → "short" (betting on economic weakness)
- "Will there be a recession?" → "short" (betting on economic downturn)
- Sports/entertainment outcomes are "long" by default (neutral exposure)
- Elections: "Will X win?" → "long" (betting on X winning, politically neutral)
- Price targets: "Will BTC hit 100K?" → "long" (betting on price increase)
- "Will X be above/over Y?" → "long" if Y is a positive threshold
- "Will X be below/under Y?" → "short" if measuring decline

Additional thematic rules:
- Ceasefires, truces, diplomatic deals, normalization, recognition, peace, shipping reopening:
  usually "risk_down"
- Strikes, invasions, attacks, clashes, military escalation, shipping disruption, supply disruption:
  usually "risk_up"
- Recession, slowdown, unemployment rise, contraction, bankruptcies:
  usually "growth_down"
- AI capability gains, adoption milestones, product launches, stronger output, economic upside:
  usually "growth_up"
- Sports, entertainment, celebrity, and non-macro markets:
  usually "neutral"

The key question is: if the YES token resolves true, is that YES outcome risk-up,
risk-down, growth-up, growth-down, neutral, or ambiguous? Return the best fit."""

USER_PROMPT_TEMPLATE = """Classify these prediction markets. For each, return the exposure of buying YES.

Markets:
{markets_text}

Return a JSON object with a single key "results" whose value is an array with one object per market:
{{"results":[{{"market_id":"...","exposure_direction":"long" or "short","yes_outcome_polarity":"risk_up|risk_down|growth_up|growth_down|neutral|ambiguous","exposure_description":"...","yes_outcome_reason":"...","confidence":0.0-1.0}}]}}

Return ONLY valid JSON."""


def _normalize_outcome_polarity(value: object) -> str:
    raw = str(value or "").strip().lower().replace(" ", "_")
    aliases = {
        "risk-positive": "risk_up",
        "risk-positive_outcome": "risk_up",
        "risk_negative": "risk_down",
        "risk-negative": "risk_down",
        "deescalation": "risk_down",
        "de_escalation": "risk_down",
        "stabilization": "risk_down",
        "stabilisation": "risk_down",
        "growth-positive": "growth_up",
        "growth_negative": "growth_down",
        "downside": "growth_down",
        "upside": "growth_up",
    }
    norm = aliases.get(raw, raw)
    if norm in VALID_OUTCOME_POLARITIES:
        return norm
    return "ambiguous"


def _heuristic_outcome_polarity(title: str) -> str:
    text = str(title or "").lower()
    if not text:
        return "ambiguous"
    risk_down_patterns = [
        r"\bceasefire\b", r"\btruce\b", r"\bpeace\b", r"\bdeal\b", r"\bnormalize\b",
        r"\bnormalise\b", r"\brecognition\b", r"\breopen\b", r"\bresume shipping\b",
        r"\bde.?escalat", r"\bstabiliz", r"\bstabilis",
    ]
    risk_up_patterns = [
        r"\bstrike\b", r"\battack\b", r"\bwar\b", r"\bmissile\b", r"\binva",
        r"\bmilitary clash\b", r"\bclash\b", r"\braid\b", r"\bclose the strait\b",
        r"\bshipping disrupt", r"\bsupply disrupt", r"\bopec cut\b", r"\bproduction cut\b",
        r"\bterminal hit\b", r"\brefinery hit\b", r"\bpipeline attack\b",
    ]
    growth_down_patterns = [
        r"\brecession\b", r"\bslowdown\b", r"\bunemployment rise\b", r"\bcontraction\b",
        r"\bdefault\b", r"\bbankrupt", r"\bshutdown\b", r"\bdecline\b", r"\bfall below\b",
        r"\bdrop below\b", r"\bund?er\b",
    ]
    growth_up_patterns = [
        r"\bai\b", r"\bmodel\b", r"\badoption\b", r"\blaunch\b", r"\brelease\b",
        r"\bgrow", r"\bincrease\b", r"\brise\b", r"\bexceed\b", r"\bsurpass\b",
        r"\babove\b", r"\bover\b", r"\bhit high\b",
    ]
    for pattern in risk_down_patterns:
        if re.search(pattern, text):
            return "risk_down"
    for pattern in risk_up_patterns:
        if re.search(pattern, text):
            return "risk_up"
    for pattern in growth_down_patterns:
        if re.search(pattern, text):
            return "growth_down"
    for pattern in growth_up_patterns:
        if re.search(pattern, text):
            return "growth_up"
    return "neutral"


def _fallback_classification(market_id: str, title: str) -> dict:
    polarity = _heuristic_outcome_polarity(title)
    exposure_direction = "short" if polarity in {"risk_up", "growth_down"} else "long"
    return {
        "market_id": market_id,
        "exposure_direction": exposure_direction,
        "yes_outcome_polarity": polarity,
        "exposure_description": f"Fallback: {title[:80]}",
        "yes_outcome_reason": "heuristic_fallback",
        "confidence": 0.0,
    }


def _load_cache() -> dict:
    """Load cached classifications."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    """Save classifications cache."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def classify_batch_llm(
    markets: list[dict],
    client: openai.OpenAI,
    *,
    model: str | None = None,
) -> list[dict]:
    """Classify a batch of markets using the configured GPT-4 class model.
    
    Args:
        markets: List of {"market_id": ..., "title": ...}
        client: OpenAI client
        
    Returns:
        List of {"market_id", "exposure_direction", "yes_outcome_polarity",
        "exposure_description", "yes_outcome_reason", "confidence"}
    """
    markets_text = "\n".join(
        f'{i+1}. [ID: {m["market_id"]}] "{m["title"]}"'
        for i, m in enumerate(markets)
    )
    
    try:
        response = client.chat.completions.create(
            model=model or DEFAULT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(markets_text=markets_text)},
            ],
            response_format={"type": "json_object"},
        )
        
        content = response.choices[0].message.content
        parsed = json.loads(content)
        
        # Handle both {"results": [...]} and direct array
        if isinstance(parsed, dict):
            results = parsed.get("results", parsed.get("markets", parsed.get("classifications", [])))
            if not results:
                # Try first list value
                for v in parsed.values():
                    if isinstance(v, list):
                        results = v
                        break
        elif isinstance(parsed, list):
            results = parsed
        else:
            results = []
            
        return results
        
    except Exception as e:
        logger.error(f"LLM classification failed: {e}")
        # Return defaults
        return [
            _fallback_classification(m["market_id"], m["title"])
            for m in markets
        ]


def _process_batch(args):
    """Process a single batch (for use with ThreadPoolExecutor)."""
    batch_markets, client, model = args
    results = classify_batch_llm(batch_markets, client, model=model)
    parsed = {}
    results_by_id = {r["market_id"]: r for r in results if "market_id" in r}
    for m in batch_markets:
        mid = m["market_id"]
        if mid in results_by_id:
            r = results_by_id[mid]
            parsed[mid] = {
                "exposure_direction": str(r.get("exposure_direction", "long")).lower(),
                "yes_outcome_polarity": _normalize_outcome_polarity(r.get("yes_outcome_polarity", "")),
                "exposure_description": str(r.get("exposure_description", "")),
                "yes_outcome_reason": str(r.get("yes_outcome_reason", "")),
                "confidence": float(r.get("confidence", 0.5) or 0.5),
                "model": str(r.get("model", model or DEFAULT_MODEL)),
            }
        else:
            parsed[mid] = _fallback_classification(mid, m["title"])
    return parsed


def classify_all_markets(
    markets_df,
    title_col: str = "title",
    id_col: str = "market_id",
    batch_size: int = 30,
    max_workers: int = 10,
    force_reclassify: bool = False,
    model: str | None = None,
) -> dict:
    """Classify all markets using the OpenAI API with caching and concurrency.
    
    Args:
        markets_df: DataFrame with market_id and title columns
        title_col: Column name for market title
        id_col: Column name for market ID
        batch_size: Markets per API call
        max_workers: Concurrent API calls
        force_reclassify: If True, ignore cache
        
    Returns:
        Dict of {market_id: {"exposure_direction", "yes_outcome_polarity",
        "exposure_description", "yes_outcome_reason", "confidence"}}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import sys
    
    cache = {} if force_reclassify else _load_cache()
    
    # Find unclassified markets
    all_markets = markets_df[[id_col, title_col]].drop_duplicates(subset=[id_col])
    unclassified = all_markets[~all_markets[id_col].isin(cache)]
    
    if len(unclassified) == 0:
        logger.info(f"All {len(all_markets)} markets already classified (cached)")
        return cache

    logger.info(f"Classifying {len(unclassified)} markets ({len(cache)} cached) with {max_workers} workers")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; falling back to heuristic exposure classification")
        for _, row in unclassified.iterrows():
            cache[str(row[id_col])] = _fallback_classification(str(row[id_col]), str(row[title_col]))
        _save_cache(cache)
        return cache

    client = openai.OpenAI(api_key=api_key)
    
    # Build batch list
    batch_markets_list = []
    for i in range(0, len(unclassified), batch_size):
        batch = unclassified.iloc[i:i+batch_size]
        batch_markets_list.append([
            {"market_id": row[id_col], "title": row[title_col]}
            for _, row in batch.iterrows()
        ])
    
    total_batches = len(batch_markets_list)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_batch, (batch, client, model or DEFAULT_MODEL)): idx
            for idx, batch in enumerate(batch_markets_list)
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                cache.update(result)
                completed += 1
                
                if completed % 20 == 0 or completed == total_batches:
                    _save_cache(cache)
                    logger.info(f"  {completed}/{total_batches} batches done ({len(cache)} total)")
                    sys.stdout.flush(); sys.stderr.flush()
            except Exception as e:
                logger.error(f"Batch failed: {e}")
                completed += 1
    
    _save_cache(cache)
    logger.info(f"Classification complete: {len(cache)} markets")
    return cache


def detect_side_batch(
    markets_df,
    title_col: str = "title",
    outcomes_col: Optional[str] = None,
    id_col: str = "market_id",
    force_reclassify: bool = False,
    model: str | None = None,
) -> "pd.DataFrame":
    """Add LLM-based side/exposure columns to a markets DataFrame.

    Adds columns:
        - token_side: YES (always for binary)
        - phrasing_polarity: positive/negative (from LLM)
        - exposure_direction: long/short (from LLM)
        - exposure_description: human-readable description
        - exposure_confidence: LLM confidence score
        - yes_outcome_polarity: theme-relative meaning of YES
        - yes_outcome_reason: short explanation of YES polarity
        - normalized_direction: 1.0 (long) or -1.0 (short)

    Returns:
        DataFrame with new columns added.
    """
    import pandas as pd

    df = markets_df.copy()
    
    # Run LLM classification
    cache = classify_all_markets(
        df,
        title_col=title_col,
        id_col=id_col,
        force_reclassify=force_reclassify,
        model=model,
    )
    
    # Map results
    df["exposure_direction"] = df[id_col].map(
        lambda mid: cache.get(mid, {}).get("exposure_direction", "long")
    )
    df["exposure_description"] = df[id_col].map(
        lambda mid: cache.get(mid, {}).get("exposure_description", "")
    )
    df["exposure_confidence"] = df[id_col].map(
        lambda mid: cache.get(mid, {}).get("confidence", 0.0)
    )
    df["yes_outcome_polarity"] = df[id_col].map(
        lambda mid: _normalize_outcome_polarity(cache.get(mid, {}).get("yes_outcome_polarity", ""))
    )
    df["yes_outcome_reason"] = df[id_col].map(
        lambda mid: cache.get(mid, {}).get("yes_outcome_reason", "")
    )
    df["direction_model"] = df[id_col].map(
        lambda mid: cache.get(mid, {}).get("model", DEFAULT_MODEL)
    )
    
    # Derived columns
    df["phrasing_polarity"] = df["exposure_direction"].map(
        {"long": "positive", "short": "negative"}
    ).fillna("positive")
    
    df["token_side"] = "YES"
    if outcomes_col and outcomes_col in df.columns:
        # For categorical markets, use the outcome name
        df.loc[df[outcomes_col].apply(lambda x: isinstance(x, list) and len(x) > 2), "token_side"] = (
            df.loc[df[outcomes_col].apply(lambda x: isinstance(x, list) and len(x) > 2), outcomes_col]
            .apply(lambda x: x[0] if isinstance(x, list) else "YES")
        )
    
    df["normalized_direction"] = df["exposure_direction"].map({"long": 1.0, "short": -1.0})
    
    polarity_counts = df["exposure_direction"].value_counts()
    logger.info(f"LLM side detection: {dict(polarity_counts)}")

    return df


def detect_token_side(
    title: str,
    outcomes: Optional[list] = None,
    tracked_token_index: int = 0,
) -> str:
    """Detect which token side is being tracked.
    
    For binary markets, returns "YES" or "NO".
    For categorical markets, returns the outcome name.
    """
    if outcomes and len(outcomes) > 2:
        return outcomes[tracked_token_index]
    if outcomes and tracked_token_index == 1:
        return "NO"
    return "YES"


def compute_exposure_direction(polarity: str, token_side: str) -> str:
    """Compute exposure direction from phrasing polarity and token side.
    
    Rules:
    - positive + YES = long (good thing happens)
    - positive + NO = short (good thing doesn't happen)  
    - negative + YES = short (bad thing happens)
    - negative + NO = long (bad thing doesn't happen)
    - neutral + YES = long (default)
    """
    if polarity == "positive":
        return "long" if token_side == "YES" else "short"
    elif polarity == "negative":
        return "short" if token_side == "YES" else "long"
    else:  # neutral
        return "long" if token_side in ("YES", ) else "short"


# Keep legacy function for backward compatibility
def detect_phrasing_polarity(title: str) -> str:
    """Legacy regex-based polarity detection. Use detect_side_batch for LLM-based."""
    import re
    
    if not title:
        return "neutral"
    
    title_lower = title.lower()
    
    # Double negation patterns (negative of negative = positive)
    DOUBLE_NEG_PATTERNS = [
        r"\bnot\s+(?:fall|drop|decline|decrease|lose|fail|collapse)",
        r"\bwon'?t\s+(?:fall|drop|decline|decrease|lose|fail|collapse)",
        r"\bwon'?t\s+\w+\s+(?:fall|drop|decline|decrease|below)",
    ]
    for p in DOUBLE_NEG_PATTERNS:
        if re.search(p, title_lower):
            return "positive"
    
    # Positive patterns
    POSITIVE_PATTERNS = [
        r"\babove\b", r"\bover\b", r"\bexceed", r"\bsurpass",
        r"\bgrow", r"\brise\b", r"\bincrease\b", r"\bwin\b",
    ]
    
    # Negative patterns
    NEGATIVE_PATTERNS = [
        r"\bnot\b", r"\bwon'?t\b", r"\bfail", r"\bnever\b",
        r"\bbelow\b", r"\bunder\b", r"\brecession",
        r"\bshutdown\b", r"\bvetoed?\b", r"\bfall\b", r"\bdecline",
        r"\bdrop\b", r"\bcollapse", r"\bdefault\b",
    ]
    
    # Check negatives first (more specific)
    for p in NEGATIVE_PATTERNS:
        if re.search(p, title_lower):
            return "negative"
    
    # Check positives
    for p in POSITIVE_PATTERNS:
        if re.search(p, title_lower):
            return "positive"
    
    return "positive"
