# Itô Markets Basket Methodology

This document describes the live static basket workflow implemented in:

- `/Users/alejandrosoumah/Documents/New project/basket-engine-ui-redesign/src/prediction_basket/thematic_baskets.py`
- `/Users/alejandrosoumah/Documents/New project/basket-engine-ui-redesign/src/prediction_basket/thematic_nav.py`

It is written for the actual platform research surface, not as a generic design note.

## 1. Operating objective

The engine builds thematic prediction-market baskets that can be inspected through one static research interface.

The workflow is intentionally narrow:

- load processed market inputs
- build monthly basket compositions
- compute daily tradable NAV
- export canonical CSVs
- publish a static analysis site tied to the same artifacts

This is not a multi-strategy framework. It is a single maintained workflow.

## 2. Core output model

The system produces two kinds of outputs.

### Canonical CSV layer
This is the source of truth.

Primary artifacts:

- `final_basket_list.csv`
- `last_year_monthly_compositions.csv`
- `basket_level_monthly.csv`
- `aggregate_basket_level.csv`
- `basket_inception_policy.csv`
- `monthly_cash_positions.csv`
- `rebalance_transitions.csv`
- `rebalance_cost_model.csv`
- `contract_lifecycle_events.csv`
- `factor_exposure_monthly.csv`
- `ticker_chain_registry.csv`
- `ticker_chain_history.csv`
- `slot_coverage_diagnostics.csv`

### Static site layer
This is the product surface built from the same CSV bundle.

- `website/index.html`: dashboard and basket-level path analysis
- `website/explorer.html`: month-by-month basket inspection
- `website/baskets.html`: continuous contract archive
- `website/methodology.html`: rulebook and artifact reference

## 3. Required inputs

The engine reads from `data/processed/`:

- `ticker_mapping.parquet`
- `ticker_chains.json`
- `polymarket_market_history.parquet`
- `prices.parquet`
- `returns.parquet`

It also reads:

- `config/basket_inception_overrides.yml`

If any of these are stale, the site may still build, but the basket outputs can become sparse, cash-heavy, or misleading.

## 4. Market objects

### Contract
A single tradable market row identified by `market_id`.

### Ticker chain
A recurring dated market family identified by `ticker_id`.

### Event family
A broader normalized event grouping used to avoid stacking dated variants of the same underlying event.

### Basket
A named thematic sleeve with deterministic rules for eligibility, selection, sizing, and cash handling.

## 5. Selection workflow

At each rebalance date, the basket builder:

1. loads the processed market universe
2. filters contracts to what existed as of that date
3. applies thematic include and exclude rules
4. determines directionality (`Long YES` or `Long NO`)
5. removes redundant chain / event / exclusivity overlaps
6. sizes selected contracts and inserts the cash sleeve

The engine is deterministic for a fixed input set.

## 6. Temporal realism

Strict temporal realism is the default.

A contract is only eligible if:

- it was already listed at the decision date
- it is not already inactive
- it passes the basket’s expiry and concentration rules

This prevents the engine from using contracts that only appeared later in history.

## 7. Directionality and risk-up semantics

Each selected contract carries explicit direction metadata.

Published fields include:

- `position_instruction`
- `market_yes_price`
- `effective_risk_price`
- `direction_reason`

Interpretation:

- `Long YES`: the basket benefits when the market’s YES outcome becomes more likely
- `Long NO`: the basket benefits when the market’s YES outcome becomes less likely because the market itself is phrased as de-escalation, normalization, or peace

Example:

- market: `Will Israel and Saudi Arabia normalize relations by March 31?`
- raw `Market YES`: low probability
- basket may hold `Long NO`
- published `Risk Price` then becomes `1 - Market YES`

The site should always display that explicitly so the user does not have to infer direction from raw market text.

## 8. Tradable NAV

The published basket line is tradable NAV.

That means:

- it starts at `100`
- it compounds actual held-position outcomes
- it includes cash, exits, and turnover effects
- it is path-dependent by design

This answers one question only:

`What money would I have made by holding this basket?`

It is not a generic topic score.

## 9. Empty spans and graph-only bridging

Some baskets can enter windows where the real basket is fully in cash and fully unpriced.

That can happen when:

- the selected basket-month is cash-only
- selected contracts have no usable price coverage
- contracts die mid-window and there is no priced replacement in the current logic

For those spans, the dashboard may show a graph-only stochastic bridge.

Important constraint:

- the bridge exists only in website payload data
- the canonical NAV in `basket_level_monthly.csv` is not changed

So the chart can remain visually readable without falsifying the underlying CSV series.

## 10. Inception policy

A basket’s raw history can begin before it is meaningfully investable.

The website therefore defaults to an effective inception date.

Automatic rule:

- `price_coverage_share >= 0.60`
- `cash_weight <= 0.25`

Fallback:

- first row where `cash_weight < 0.95`

Manual overrides live in `config/basket_inception_overrides.yml`.

The site defaults to `Since Inception`, but `Full History` remains available.

## 11. Cash and lifecycle handling

Cash is a real sleeve, not just a display artifact.

Lifecycle events include:

- new selection
- expiry or resolution exit
- certainty-threshold exit when priced logic is active
- rebalance replacement

Lifecycle reasons are written to `contract_lifecycle_events.csv`.

Monthly cash context is written to `monthly_cash_positions.csv`.

## 12. Chain lineage and contract archive

The platform research workflow depends on chain continuity.

The engine therefore exports:

- `ticker_chain_registry.csv`
- `ticker_chain_history.csv`

These outputs are used by both the explorer and the basket archive pages so a user can see:

- which dated market was selected in a given month
- what other chain members existed
- how the chain evolved over time

## 13. Rebalance and turnover diagnostics

The engine publishes:

- `rebalance_transitions.csv`
- `rebalance_cost_model.csv`

These are used to inspect:

- how many entries and exits were made
- turnover burden
- estimated execution drag

## 14. UI structure and intended use

### Dashboard
Use for:
- basket-level path analysis
- warnings and coverage diagnostics
- aggregate context
- quick access to canonical files

### Explorer
Use for:
- exact basket month inspection
- contract-by-contract review for a selected rebalance
- direction and chain review of one selected contract

### Baskets
Use for:
- continuous archive of every contract used by a basket
- average and peak weight review
- long-range composition research

### Methodology
Use for:
- operating rulebook
- output map
- implementation semantics

## 15. Run command

```bash
cd '/Users/alejandrosoumah/Documents/New project/basket-engine-ui-redesign'
python -m src.prediction_basket.thematic_baskets --start 2022-01-01 --end 2026-03-01 --prune-unused --log-progress
```

Useful flags:

- `--no-site`
- `--prune-unused`
- `--allow-missing-temporal-history`
- `--no-strict-temporal`
- `--refresh-exposure-directions`
- `--force-exposure-refresh`
- `--exposure-model`
- `--log-progress`

## 16. Current operating constraint

This redesign lives in an isolated sibling workspace because the parent checkout has no usable commit baseline for a real git worktree.

That isolation preserves the current production-like basket-engine folder while allowing source-level redesign work here.
