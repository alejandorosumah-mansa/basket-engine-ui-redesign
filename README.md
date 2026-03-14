# Ito Markets Basket Engine

Static research engine for thematic prediction-market baskets.

Takes processed market inputs, builds deterministic basket compositions and tradable NAV series, and publishes a static analysis site for the platform research workflow.

## Architecture

```
                           ┌──────────────────────────────────┐
                           │        Canonical Inputs          │
                           │    data/processed/               │
                           │  ┌────────────────────────────┐  │
                           │  │ ticker_mapping.parquet      │  │
                           │  │ ticker_chains.json          │  │
                           │  │ polymarket_market_history    │  │
                           │  │   .parquet                  │  │
                           │  │ prices.parquet              │  │
                           │  │ returns.parquet             │  │
                           │  └────────────────────────────┘  │
                           │  config/                         │
                           │  └─ basket_inception_overrides   │
                           │       .yml                       │
                           └──────────────┬───────────────────┘
                                          │
                    ┌─────────────────────────────────────────────┐
                    │                                             │
                    ▼                                             ▼
   ┌────────────────────────────────┐        ┌───────────────────────────────┐
   │   Market Universe Loader      │        │   Exposure / Side Detection   │
   │                                │        │   src/exposure/               │
   │  Load & normalize contracts    │        │   side_detection.py           │
   │  from parquet files or DB.     │        │                               │
   │  Attach temporal metadata      │        │  LLM-based classification     │
   │  (listing date, expiry,        │        │  (OpenAI) to determine:       │
   │   resolution, lifecycle).      │        │  - Long YES vs Long NO        │
   │  Resolve ticker chains for     │        │  - risk_up / risk_down        │
   │  continuous contract lineage.  │        │  - growth_up / growth_down    │
   │                                │        │  Results cached in            │
   │  Optional: --db-url reads     │        │  exposure_classifications     │
   │  from PostgreSQL via db_io.py  │        │  .json                        │
   └──────────────┬─────────────────┘        └──────────────┬────────────────┘
                  │                                         │
                  └──────────────┬──────────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────────────┐
          │          Thematic Basket Builder                     │
          │          src/prediction_basket/thematic_baskets.py   │
          │                                                      │
          │  For each rebalance date:                            │
          │                                                      │
          │  1. Filter   ── Apply thematic include/exclude       │
          │                  patterns and global exclusions       │
          │                  (elections, sports, entertainment)   │
          │                                                      │
          │  2. Match    ── Score contracts against basket        │
          │                  themes using keyword + semantic      │
          │                  heuristics                           │
          │                                                      │
          │  3. Select   ── Pick top contracts per slot with      │
          │                  tenor, probability, and event-       │
          │                  family deduplication constraints     │
          │                                                      │
          │  4. Direct   ── Assign Long YES / Long NO per        │
          │                  contract via side detection          │
          │                                                      │
          │  5. Weight   ── Size positions with slot caps,       │
          │                  max-weight clipping, and explicit    │
          │                  cash sleeve insertion                │
          │                                                      │
          │  Basket specs: ~13 thematic baskets (AI, Conflict,   │
          │  Energy, Governance, Supply Chain, Climate, etc.)     │
          └──────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────────────┐
          │          Tradable NAV Engine                         │
          │          src/prediction_basket/thematic_nav.py       │
          │                                                      │
          │  Build daily NAV series per basket:                  │
          │  - Start at 100, compound held-position outcomes     │
          │  - Include cash, exits, turnover effects             │
          │  - Certainty exits at p >= 0.95 or p <= 0.05        │
          │  - Fee, spread, and slippage modeling                │
          │  - Binary settlement at resolution                   │
          │                                                      │
          │  The NAV answers: "What money would I have made      │
          │  by holding this basket?"                            │
          └──────────────────────┬───────────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
┌──────────────────────────────┐   ┌─────────────────────────────────────┐
│   Canonical CSV Outputs      │   │   Static Research Website           │
│   data/outputs/              │   │   data/outputs/.../website/         │
│   prediction_basket/         │   │                                     │
│                              │   │  ┌───────────┐  ┌───────────────┐  │
│  final_basket_list.csv       │   │  │ Dashboard │  │   Explorer    │  │
│  basket_level_monthly.csv    │   │  │ index     │  │   explorer    │  │
│  aggregate_basket_level.csv  │   │  │ .html     │  │   .html       │  │
│  monthly_cash_positions.csv  │   │  └───────────┘  └───────────────┘  │
│  rebalance_transitions.csv   │   │  ┌───────────┐  ┌───────────────┐  │
│  rebalance_cost_model.csv    │   │  │  Baskets  │  │  Methodology  │  │
│  contract_lifecycle_events   │   │  │  baskets  │  │  methodology  │  │
│    .csv                      │   │  │  .html    │  │  .html        │  │
│  factor_exposure_monthly.csv │   │  └───────────┘  └───────────────┘  │
│  ticker_chain_registry.csv   │   │                                     │
│  ticker_chain_history.csv    │   │  Built from the same CSV outputs.  │
│  slot_coverage_diagnostics   │   │  Self-contained (ECharts bundled). │
│    .csv                      │   │  No backend required.              │
│  + 3 more                    │   │                                     │
└──────────────────────────────┘   └─────────────────────────────────────┘
```

## What this repo produces

| Output | Location | Purpose |
|--------|----------|---------|
| Canonical CSVs | `data/outputs/prediction_basket/*.csv` | Source-of-truth basket compositions, NAV series, cost models, lifecycle events |
| Static website | `data/outputs/prediction_basket/website/` | Research surface for basket inspection, contract audit, and methodology review |
| Methodology doc | `docs/prediction_basket_methodology.md` | Operating rulebook for the basket workflow |

The website is not a mockup. It is generated from the same CSV outputs and is the intended research surface.

## Repository layout

```text
basket-creation-engine/
├── config/
│   └── basket_inception_overrides.yml   # Manual inception date overrides
├── data/
│   ├── processed/                       # Canonical inputs (parquet + JSON)
│   └── outputs/
│       └── prediction_basket/
│           ├── *.csv                    # Canonical CSV artifacts
│           └── website/                 # Static research site
├── docs/
│   └── prediction_basket_methodology.md # Full methodology reference
├── src/
│   ├── exposure/
│   │   └── side_detection.py            # LLM-based direction classification
│   └── prediction_basket/
│       ├── assets/
│       │   └── echarts.min.js           # Bundled chart library
│       ├── db_io.py                     # Optional DB I/O (PostgreSQL)
│       ├── thematic_baskets.py          # Basket builder + site generator
│       └── thematic_nav.py             # Tradable NAV construction
├── tests/
│   ├── test_exposure.py
│   ├── test_thematic_baskets.py
│   └── test_thematic_nav.py
├── pytest.ini
└── requirements.txt
```

## Core concepts

### Tradable NAV, not topic scores

The published basket line is **tradable NAV**: it starts at 100, compounds actual held-position outcomes including cash, exits, and turnover. It is path-dependent and answers one question: *"What money would I have made by holding this basket?"*

### Temporal realism

Contracts are only eligible if they were already listed at the decision date and are not yet inactive. This prevents look-ahead bias in historical runs.

### Direction mapping

Each contract carries explicit direction metadata (`Long YES` or `Long NO`) determined by LLM-based classification. A `Long NO` position means the basket benefits when the YES outcome becomes *less* likely -- used when the market is phrased as de-escalation or normalization.

### Inception policy

Baskets default to an effective inception date where `price_coverage >= 60%` and `cash_weight <= 25%`. Manual overrides live in `config/basket_inception_overrides.yml`. The site defaults to "Since Inception" but "Full History" is always available.

### Cash sleeve

Cash is a real position sleeve, not a display artifact. It is explicitly tracked in compositions and NAV calculations.

## Quick start

### Prerequisites

- Python 3.11+
- Canonical input files in `data/processed/` (parquet + JSON)
- OpenAI API key (for direction classification, optional if cache exists)

### Install

```bash
pip install -r requirements.txt
```

### Run the engine

```bash
python -m src.prediction_basket.thematic_baskets \
  --start 2022-01-01 \
  --end 2026-03-01 \
  --prune-unused
```

This builds all basket compositions, computes NAV series, exports CSVs, and generates the static website.

### CLI flags

| Flag | Description |
|------|-------------|
| `--start` / `--end` | Date range for the historical run |
| `--no-site` | Skip website generation |
| `--prune-unused` | Remove non-canonical files from the output folder |
| `--allow-missing-temporal-history` | Weaken strict temporal validation |
| `--no-strict-temporal` | Disable as-of listing-date enforcement |
| `--refresh-exposure-directions` | Refresh direction cache before generation |
| `--force-exposure-refresh` | Ignore saved exposure cache entirely |
| `--exposure-model` | Override the OpenAI model for direction classification |
| `--log-progress` | Emit build and NAV progress logs during long runs |
| `--db-url` | Read market data from PostgreSQL instead of parquet files |

### Run tests

```bash
pytest -q
```

## Product surfaces

The static site has four pages, each built from the canonical CSV outputs:

### Dashboard (`index.html`)
Basket-level tradable NAV paths, aggregate system context, coverage and warning diagnostics, and direct links to CSV artifacts.

### Explorer (`explorer.html`)
Month-by-month basket composition inspector. Shows exact contract selection, direction mapping (`Long YES` / `Long NO`), and ticker-chain lineage for each contract.

### Baskets (`baskets.html`)
Full contract archive by basket. Average and peak weight history, continuous contract review across the run window, and composition history with chain transitions.

### Methodology (`methodology.html`)
Operating rulebook: data dependencies, rebalance logic, lifecycle rules, inception policy, and the complete artifact map.

## Canonical inputs

The engine depends on these processed inputs:

| File | Purpose |
|------|---------|
| `data/processed/ticker_mapping.parquet` | Market universe with contract metadata |
| `data/processed/ticker_chains.json` | Ticker chain definitions for continuous contracts |
| `data/processed/polymarket_market_history.parquet` | Lifecycle events (listing, resolution, closure) |
| `data/processed/prices.parquet` | Daily close prices per contract |
| `data/processed/returns.parquet` | Daily return series |
| `config/basket_inception_overrides.yml` | Manual inception date overrides per basket |

If any input is stale or incomplete, the site may still render but basket quality will degrade.

## Canonical outputs

| CSV | Content |
|-----|---------|
| `final_basket_list.csv` | Master basket definitions |
| `last_year_monthly_compositions.csv` | Trailing 12-month compositions |
| `basket_level_monthly.csv` | Monthly NAV and stats per basket |
| `aggregate_basket_level.csv` | System-wide aggregate performance |
| `basket_inception_policy.csv` | Effective inception dates |
| `basket_monthly_summary.csv` | Monthly basket summary statistics |
| `monthly_cash_positions.csv` | Cash sleeve tracking |
| `rebalance_transitions.csv` | Contract entries and exits at rebalance |
| `rebalance_cost_model.csv` | Slippage, fees, and transaction costs |
| `contract_lifecycle_events.csv` | Expiry, resolution, certainty exits |
| `factor_exposure_monthly.csv` | Risk factor exposure per basket |
| `ticker_chain_registry.csv` | Ticker chain metadata |
| `ticker_chain_history.csv` | Historical chain transitions |
| `slot_coverage_diagnostics.csv` | Slot fill rates and coverage warnings |

## Design constraints

- The published basket line is tradable NAV, not a generic topic score.
- Empty spans (full-cash, fully unpriced) may be visually bridged in the chart layer only; canonical NAV CSVs are never modified.
- The website is static and must remain readable without a backend.
- The methodology page is part of the product surface, not an internal note.
- The engine is deterministic for a fixed input set.

## Dependencies

```
pandas >= 2.0        # DataFrames and time series
pyarrow >= 14.0      # Parquet I/O
numpy >= 1.24        # Numerical operations
matplotlib >= 3.8    # Basket composition charts
openai >= 1.0        # Direction classification (side detection)
pyyaml >= 6.0        # Config file parsing
pytest >= 7.0        # Test suite
```

ECharts is bundled locally (`src/prediction_basket/assets/echarts.min.js`) to keep the site fully self-contained.

## Module reference

| Module | Role |
|--------|------|
| `src/prediction_basket/thematic_baskets.py` | Main orchestrator: basket spec definitions, thematic filtering, contract selection, weighting, CSV export, and full static site generation (~7900 lines) |
| `src/prediction_basket/thematic_nav.py` | Tradable NAV construction: daily mark-to-market, cash handling, certainty exits, fee modeling, binary settlement |
| `src/prediction_basket/db_io.py` | Optional database I/O layer: reads market universe and prices from PostgreSQL, writes basket results back to DB |
| `src/exposure/side_detection.py` | LLM-based direction classification: determines `Long YES` vs `Long NO` per contract using OpenAI with local cache |
