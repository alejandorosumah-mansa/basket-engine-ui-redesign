from __future__ import annotations

import pandas as pd

import src.exposure.side_detection as side_detection
from src.exposure.side_detection import (
    _fallback_classification,
    _normalize_outcome_polarity,
    compute_exposure_direction,
    detect_phrasing_polarity,
    detect_side_batch,
    detect_token_side,
)


def test_normalize_outcome_polarity_maps_aliases():
    assert _normalize_outcome_polarity('deescalation') == 'risk_down'
    assert _normalize_outcome_polarity('risk-positive') == 'risk_up'
    assert _normalize_outcome_polarity('upside') == 'growth_up'
    assert _normalize_outcome_polarity('unknown-value') == 'ambiguous'


def test_fallback_classification_sets_direction_from_polarity():
    conflict = _fallback_classification('m1', 'Will Israel strike Iran?')
    recession = _fallback_classification('m2', 'Will there be a recession in 2026?')
    ai = _fallback_classification('m3', 'Will OpenAI release a stronger model this year?')

    assert conflict['yes_outcome_polarity'] == 'risk_up'
    assert conflict['exposure_direction'] == 'short'
    assert recession['yes_outcome_polarity'] == 'growth_down'
    assert recession['exposure_direction'] == 'short'
    assert ai['yes_outcome_polarity'] == 'growth_up'
    assert ai['exposure_direction'] == 'long'


def test_detect_side_batch_adds_current_llm_columns(monkeypatch):
    monkeypatch.setattr(
        side_detection,
        'classify_all_markets',
        lambda *args, **kwargs: {
            'm_risk': {
                'exposure_direction': 'short',
                'yes_outcome_polarity': 'risk_up',
                'exposure_description': 'Conflict escalation',
                'yes_outcome_reason': 'military escalation increases risk',
                'confidence': 0.94,
                'model': 'gpt-4.1-mini',
            },
            'm_growth': {
                'exposure_direction': 'long',
                'yes_outcome_polarity': 'growth_up',
                'exposure_description': 'AI upside',
                'yes_outcome_reason': 'capability advance is upside',
                'confidence': 0.91,
                'model': 'gpt-4.1-mini',
            },
        },
    )

    df = pd.DataFrame(
        {
            'market_id': ['m_risk', 'm_growth'],
            'title': ['Will Israel strike Iran?', 'Will OpenAI release a stronger model?'],
        }
    )

    out = detect_side_batch(df)

    assert list(out['token_side']) == ['YES', 'YES']
    assert list(out['exposure_direction']) == ['short', 'long']
    assert list(out['yes_outcome_polarity']) == ['risk_up', 'growth_up']
    assert list(out['normalized_direction']) == [-1.0, 1.0]
    assert list(out['direction_model']) == ['gpt-4.1-mini', 'gpt-4.1-mini']


def test_legacy_side_helpers_still_behave():
    assert detect_token_side('Will X happen?') == 'YES'
    assert detect_token_side('Will X happen?', ['Yes', 'No'], tracked_token_index=1) == 'NO'
    assert compute_exposure_direction('positive', 'YES') == 'long'
    assert compute_exposure_direction('negative', 'YES') == 'short'
    assert detect_phrasing_polarity('Will Bitcoin exceed $100K?') == 'positive'
    assert detect_phrasing_polarity('Will there be a recession in 2026?') == 'negative'
