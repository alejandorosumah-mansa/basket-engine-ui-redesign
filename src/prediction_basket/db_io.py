"""DB I/O for basket-engine: reads market data from DB, writes results to DB.

Replaces parquet/CSV file I/O when --db-url is provided.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


# ── Readers ──────────────────────────────────────────────────────────

def load_market_universe_from_db(db_url: str) -> pd.DataFrame:
    """Load market universe from markets table.

    Returns a DataFrame matching the schema expected by load_market_universe().
    """
    engine = create_engine(db_url)
    query = text("""
        SELECT market_id, ticker, title, expiration, platform,
               event_ticker, created_at, resolution_time, closed_time,
               resolution_value, status
        FROM markets
        WHERE market_id IS NOT NULL
    """)
    df = pd.read_sql(query, engine)
    engine.dispose()

    if df.empty:
        return pd.DataFrame(columns=[
            "market_id", "ticker_id", "ticker_name", "title",
            "end_date_parsed", "end_date", "platform", "event_slug",
            "listed_at", "inactive_at", "current_price", "temporal_source",
        ])

    df["market_id"] = df["market_id"].astype(str)
    df["ticker_id"] = df["ticker"].fillna("").astype(str)
    df["ticker_name"] = df["ticker"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["title_l"] = df["title"].str.lower()
    df["end_date_parsed"] = pd.to_datetime(df["expiration"], errors="coerce")
    df["end_date"] = df["end_date_parsed"]
    df = df.dropna(subset=["end_date"]).copy()
    df["platform"] = df["platform"].fillna("polymarket").astype(str)
    df["event_slug"] = df["event_ticker"].fillna("").astype(str)
    df["listed_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # inactive_at = earliest of resolution_time, closed_time
    res = pd.to_datetime(df["resolution_time"], errors="coerce")
    clo = pd.to_datetime(df["closed_time"], errors="coerce")
    df["inactive_at"] = pd.concat([res, clo], axis=1).min(axis=1)

    df["current_price"] = np.nan
    df["temporal_source"] = "db"

    # Classification columns the engine checks for
    df["llm_category"] = "unknown"
    df["llm_secondary_category"] = ""
    df["classification_source"] = "unknown"
    df["classification_confidence"] = 0.0

    return df.reset_index(drop=True)


def load_prices_from_db(db_url: str, market_ids: list[str] | None = None) -> pd.DataFrame:
    """Load price history from markets_historical table."""
    engine = create_engine(db_url)
    if market_ids:
        placeholders = ", ".join(f":mid_{i}" for i in range(len(market_ids)))
        query = text(f"""
            SELECT market_id, date, close_price, volume
            FROM markets_historical
            WHERE market_id IN ({placeholders})
            ORDER BY market_id, date
        """)
        params = {f"mid_{i}": mid for i, mid in enumerate(market_ids)}
        df = pd.read_sql(query, engine, params=params)
    else:
        query = text("""
            SELECT market_id, date, close_price, volume
            FROM markets_historical
            ORDER BY market_id, date
        """)
        df = pd.read_sql(query, engine)
    engine.dispose()

    if df.empty:
        return pd.DataFrame(columns=["market_id", "date", "close_price", "volume"])

    df["market_id"] = df["market_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    return df.reset_index(drop=True)


def load_market_lifecycle_from_db(
    db_url: str, market_ids: list[str] | None = None
) -> pd.DataFrame:
    """Load lifecycle data from markets table.

    Returns DataFrame with columns matching load_market_lifecycle():
    [market_id, created_date, end_date, inactive_date, resolution_date, resolution_value]
    """
    engine = create_engine(db_url)
    if market_ids:
        placeholders = ", ".join(f":mid_{i}" for i in range(len(market_ids)))
        query = text(f"""
            SELECT market_id, created_at, expiration, resolution_time,
                   closed_time, resolution_value, status
            FROM markets
            WHERE market_id IN ({placeholders})
        """)
        params = {f"mid_{i}": mid for i, mid in enumerate(market_ids)}
        df = pd.read_sql(query, engine, params=params)
    else:
        query = text("""
            SELECT market_id, created_at, expiration, resolution_time,
                   closed_time, resolution_value, status
            FROM markets
        """)
        df = pd.read_sql(query, engine)
    engine.dispose()

    if df.empty:
        return pd.DataFrame(
            columns=["market_id", "created_date", "end_date", "inactive_date",
                      "resolution_date", "resolution_value"]
        )

    df["market_id"] = df["market_id"].astype(str)

    def _to_naive(series):
        dt = pd.to_datetime(series, errors="coerce", utc=True)
        return dt.dt.tz_convert(None).dt.normalize()

    df["created_date"] = _to_naive(df["created_at"])
    df["end_date"] = _to_naive(df["expiration"])
    df["resolution_date"] = _to_naive(df["resolution_time"])
    closed_date = _to_naive(df["closed_time"])
    df["inactive_date"] = pd.concat(
        [df["resolution_date"], closed_date, df["end_date"]], axis=1
    ).min(axis=1)

    if "resolution_value" in df.columns:
        df["resolution_value"] = pd.to_numeric(df["resolution_value"], errors="coerce")
    else:
        df["resolution_value"] = np.nan

    df = df.sort_values(["market_id", "inactive_date", "created_date"]).drop_duplicates(
        "market_id", keep="last"
    )
    return df[
        ["market_id", "created_date", "end_date", "inactive_date",
         "resolution_date", "resolution_value"]
    ].reset_index(drop=True)


def load_ticker_chains_from_db(db_url: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ticker chain data from markets table.

    Returns (chain_meta, chain_markets) DataFrames matching load_ticker_chains().
    """
    engine = create_engine(db_url)
    query = text("""
        SELECT ticker AS ticker_id, ticker AS ticker_name,
               market_id, title, event_ticker AS event_slug, expiration
        FROM markets
        WHERE ticker IS NOT NULL AND ticker != ''
        ORDER BY ticker, expiration
    """)
    df = pd.read_sql(query, engine)
    engine.dispose()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["ticker_id"] = df["ticker_id"].astype(str)
    df["ticker_name"] = df["ticker_name"].fillna("").astype(str)
    df["market_id"] = df["market_id"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["event_slug"] = df["event_slug"].fillna("").astype(str)
    df["end_date"] = pd.to_datetime(df["expiration"], errors="coerce")
    df = df[(df["ticker_id"] != "") & (df["market_id"] != "")].copy()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    markets_df = (
        df.sort_values(["ticker_id", "end_date", "market_id"])
        .drop_duplicates(["ticker_id", "market_id"], keep="last")
        .reset_index(drop=True)
    )
    chain_df = (
        markets_df.groupby(["ticker_id", "ticker_name"], dropna=False)
        .agg(
            chain_market_count=("market_id", "nunique"),
            chain_first_end_date=("end_date", "min"),
            chain_last_end_date=("end_date", "max"),
            chain_event_slugs=(
                "event_slug",
                lambda s: ", ".join(sorted({str(x).strip() for x in s if str(x).strip()})),
            ),
        )
        .reset_index()
    )
    for col in ["chain_first_end_date", "chain_last_end_date"]:
        chain_df[col] = (
            pd.to_datetime(chain_df[col], errors="coerce")
            .dt.date.astype(str)
            .replace("NaT", "")
        )

    return chain_df, markets_df


# ── Writer ───────────────────────────────────────────────────────────

def write_results_to_db(db_url: str, run_key: str, results: dict) -> dict:
    """Write engine output DataFrames to strategy_backtest_* tables.

    Args:
        db_url: Postgres connection URL.
        run_key: Unique run key for strategy_backtest_runs.
        results: Dict mapping logical names to DataFrames. Expected keys:
            basket_levels_df, aggregate_df, compositions_df,
            factor_exposure_df, ticker_registry_df, ticker_history_df,
            lifecycle_df, cost_model_df

    Returns:
        Dict with run_key and counts of rows written per table.
    """
    engine = create_engine(db_url)
    counts = {}

    with engine.begin() as conn:
        # Create or get the run
        existing = conn.execute(
            text("SELECT id FROM strategy_backtest_runs WHERE run_key = :rk"),
            {"rk": run_key},
        ).fetchone()

        if existing:
            run_id = existing[0]
            # Delete existing data for this run
            for table in [
                "strategy_backtest_basket_levels",
                "strategy_backtest_aggregate_levels",
                "strategy_backtest_constituents",
                "strategy_backtest_factor_exposures",
                "strategy_backtest_ticker_registry",
                "strategy_backtest_ticker_history",
                "strategy_backtest_lifecycle_events",
                "strategy_backtest_cost_models",
            ]:
                conn.execute(
                    text(f"DELETE FROM {table} WHERE run_id = :rid"),
                    {"rid": run_id},
                )
            conn.execute(
                text("UPDATE strategy_backtest_runs SET completed_at = :now WHERE id = :rid"),
                {"rid": run_id, "now": datetime.utcnow()},
            )
        else:
            result = conn.execute(
                text("""
                    INSERT INTO strategy_backtest_runs (run_key, source, status, created_at, completed_at)
                    VALUES (:rk, 'engine_db', 'completed', :now, :now)
                    RETURNING id
                """),
                {"rk": run_key, "now": datetime.utcnow()},
            )
            run_id = result.fetchone()[0]

        # Write each DataFrame
        table_map = {
            "basket_levels_df": "strategy_backtest_basket_levels",
            "aggregate_df": "strategy_backtest_aggregate_levels",
            "compositions_df": "strategy_backtest_constituents",
            "factor_exposure_df": "strategy_backtest_factor_exposures",
            "ticker_registry_df": "strategy_backtest_ticker_registry",
            "ticker_history_df": "strategy_backtest_ticker_history",
            "lifecycle_df": "strategy_backtest_lifecycle_events",
            "cost_model_df": "strategy_backtest_cost_models",
        }

        for key, table_name in table_map.items():
            df = results.get(key)
            if df is None or df.empty:
                counts[table_name] = 0
                continue

            df = df.copy()
            df["run_id"] = run_id

            # Handle factor exposure: consolidate beta_ columns into JSON
            if key == "factor_exposure_df":
                beta_cols = [c for c in df.columns if c.startswith("beta_")]
                if beta_cols:
                    df["exposures"] = df[beta_cols].apply(
                        lambda row: {k: float(v) for k, v in row.items() if pd.notna(v)},
                        axis=1,
                    ).apply(json.dumps)
                    df = df.drop(columns=beta_cols)

            # Convert date columns
            for col in df.columns:
                if "date" in col.lower() and col != "run_id":
                    df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

            # Drop the 'id' column if present (auto-generated)
            df = df.drop(columns=["id"], errors="ignore")

            # Only keep columns that exist in the target table
            existing_cols = {
                r[0] for r in conn.execute(
                    text("""
                        SELECT column_name FROM information_schema.columns
                        WHERE table_name = :tn
                    """),
                    {"tn": table_name},
                ).fetchall()
            }
            keep_cols = [c for c in df.columns if c in existing_cols]
            df = df[keep_cols]

            df.to_sql(table_name, conn, if_exists="append", index=False, method="multi")
            counts[table_name] = len(df)
            logger.info("Wrote %d rows to %s", len(df), table_name)

        # Update run date range from basket levels
        date_range = conn.execute(
            text("""
                SELECT MIN(rebalance_date), MAX(rebalance_date)
                FROM strategy_backtest_basket_levels
                WHERE run_id = :rid
            """),
            {"rid": run_id},
        ).fetchone()
        if date_range and date_range[0]:
            conn.execute(
                text("""
                    UPDATE strategy_backtest_runs
                    SET data_start_date = :start, data_end_date = :end
                    WHERE id = :rid
                """),
                {"rid": run_id, "start": date_range[0], "end": date_range[1]},
            )

    engine.dispose()
    return {"run_key": run_key, "run_id": run_id, "counts": counts}
