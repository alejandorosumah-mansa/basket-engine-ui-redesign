# Itô Markets Basket Engine

Static research engine for thematic prediction-market baskets.

This repository does one job: take processed market inputs, build deterministic basket compositions and tradable NAV series, and publish a static analysis site that can be used directly inside the platform workflow.

## What this repo produces

- Canonical CSV outputs in `data/outputs/prediction_basket/`
- Static website in `data/outputs/prediction_basket/website/`
- Reproducible basket methodology in `docs/prediction_basket_methodology.md`

The website is not a mockup. It is generated from the same CSV outputs and is intended to be the research surface for basket inspection, month-by-month contract audit, and methodology review.

## Repository layout

```text
basket-engine/
├── config/
│   └── basket_inception_overrides.yml
├── data/
│   ├── processed/
│   │   ├── polymarket_market_history.parquet
│   │   ├── prices.parquet
│   │   ├── returns.parquet
│   │   ├── ticker_chains.json
│   │   └── ticker_mapping.parquet
│   └── outputs/
│       └── prediction_basket/
│           ├── *.csv
│           └── website/
├── docs/
│   └── prediction_basket_methodology.md
├── src/
│   ├── exposure/
│   │   └── side_detection.py
│   └── prediction_basket/
│       ├── assets/
│       │   └── echarts.min.js
│       ├── thematic_baskets.py
│       └── thematic_nav.py
├── tests/
├── pytest.ini
└── requirements.txt
```

## Product surfaces

### 1. Dashboard
`data/outputs/prediction_basket/website/index.html`

Use this page for:
- basket-level tradable NAV paths
- aggregate system context
- basket coverage and warning diagnostics
- direct access to canonical CSV artifacts

### 2. Explorer
`data/outputs/prediction_basket/website/explorer.html`

Use this page for:
- exact basket composition on a selected rebalance month
- selected contract detail
- direction mapping (`Long YES` / `Long NO`)
- ticker-chain lineage for the selected contract

### 3. Baskets
`data/outputs/prediction_basket/website/baskets.html`

Use this page for:
- full contract archive by basket
- average and peak weight history
- continuous contract review across the whole run window
- basket composition history and chain transitions

### 4. Methodology
`data/outputs/prediction_basket/website/methodology.html`

Use this page for:
- operating rulebook
- data dependencies
- rebalance and lifecycle logic
- inception and display logic
- artifact map and appendices

## Canonical inputs

The engine depends on these processed inputs:

- `data/processed/ticker_mapping.parquet`
- `data/processed/ticker_chains.json`
- `data/processed/polymarket_market_history.parquet`
- `data/processed/prices.parquet`
- `data/processed/returns.parquet`
- `config/basket_inception_overrides.yml`

If one of these is stale or incomplete, the website may still render, but basket quality will degrade.

## Core workflow

1. Load processed market universe and temporal metadata.
2. Apply thematic filters and direction logic.
3. Build monthly basket compositions.
4. Build daily tradable NAV series.
5. Export canonical CSVs.
6. Generate the static site from the same output set.

The entrypoint is:

```bash
cd '/Users/alejandrosoumah/Documents/New project/basket-engine'
python -m src.prediction_basket.thematic_baskets --start 2022-01-01 --end 2026-03-01 --prune-unused
```

Useful flags:

- `--no-site`: skip website generation
- `--prune-unused`: remove non-canonical files from the output folder
- `--allow-missing-temporal-history`: weaken strict temporal validation
- `--no-strict-temporal`: disable as-of listing-date enforcement
- `--refresh-exposure-directions`: refresh direction cache before generation
- `--force-exposure-refresh`: ignore saved exposure cache
- `--exposure-model`: override the OpenAI model for direction classification
- `--log-progress`: emit build and NAV progress logs during a long historical run

## Canonical outputs

Primary CSVs:

- `final_basket_list.csv`
- `last_year_monthly_compositions.csv`
- `basket_level_monthly.csv`
- `aggregate_basket_level.csv`
- `basket_inception_policy.csv`
- `basket_monthly_summary.csv`
- `monthly_cash_positions.csv`
- `rebalance_transitions.csv`
- `rebalance_cost_model.csv`
- `contract_lifecycle_events.csv`
- `factor_exposure_monthly.csv`
- `ticker_chain_registry.csv`
- `ticker_chain_history.csv`
- `slot_coverage_diagnostics.csv`

Static site:

- `data/outputs/prediction_basket/website/index.html`
- `data/outputs/prediction_basket/website/explorer.html`
- `data/outputs/prediction_basket/website/baskets.html`
- `data/outputs/prediction_basket/website/methodology.html`

## Design assumptions

- The published basket line is tradable NAV, not a generic topic score.
- Empty full-cash full-unpriced spans may be visually bridged in the chart layer only; the underlying NAV CSV remains untouched.
- The website is static and must remain readable without a backend.
- The methodology page is part of the product surface, not a separate internal note.

## Development notes

- `src/prediction_basket/thematic_baskets.py` is the generator and website publisher.
- `src/prediction_basket/thematic_nav.py` handles tradable NAV construction.
- `src/exposure/side_detection.py` handles direction mapping.
- `src/prediction_basket/assets/echarts.min.js` is bundled locally to keep the site self-contained.

## Tests

```bash
cd '/Users/alejandrosoumah/Documents/New project/basket-engine'
pytest -q
```

## Current constraint

This workspace is being used as an isolated redesign copy because the parent repository has no usable commit baseline for a true git worktree. The engine and site changes are therefore contained here until you choose how to merge them back.
