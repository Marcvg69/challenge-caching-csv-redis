#!/usr/bin/env python3
"""
main.py — CSV -> Redis cached analytics

Queries:
  - avg_arr_delay_by_airline
  - flights_by_origin
  - avg_dep_delay_by_month

Flags:
  --clear-cache        Clear 'agg:' keys and exit
  --show-cache         List keys/snippets from Redis and exit
  --full               With --show-cache, pretty-print full JSON
  --limit N            Print top-N rows of the query result (default 10)
  --csv PATH           Override CSV path (else take CSV_PATH from .env)

Env (.env is auto-loaded):
  CSV_PATH             Default CSV path (e.g., /Users/Marc/Datasets/.../flights.csv)
  AIRLINES_CSV         Optional: code->name lookup for airlines
  AIRPORTS_CSV         Optional: code->name lookup for airports
  CACHE_TTL_SECONDS    TTL for cached results (default: 3600)
  CHUNKSIZE            If set (>0), enable chunked processing
  REDIS_*              Handled in app.cache
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

# ---------------------------------------------------------------------------
# Minimal .env loader (no external deps). MUST run BEFORE reading env vars.
# ---------------------------------------------------------------------------
def load_env(override: bool = False):
    for path in (os.path.join(os.getcwd(), ".env"), os.path.join(os.getcwd(), "app", ".env")):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip()
                if not override and k in os.environ:
                    continue
                os.environ[k] = v
        break

load_env(override=False)

# 🔧 IMPORTANT: use absolute package import
from app.cache import (
    key_for,
    get_json,
    set_json,
    incr,
    clear_prefix,
    list_cache,
)

# Logging
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

# Defaults (read AFTER .env is loaded)
CSV_PATH_DEFAULT = os.getenv("CSV_PATH", "data/Flights.csv")
DEFAULT_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
CHUNK_SIZE_ENV = os.getenv("CHUNKSIZE", "").strip()
PRINT_LIMIT_DEFAULT = 10

# Columns we actually need (prevents DtypeWarning & saves memory)
REQUIRED_COLS_KAGGLE = [
    "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "DEPARTURE_DELAY", "ARRIVAL_DELAY", "YEAR", "MONTH"
]
REQUIRED_COLS_BTS = [
    "OP_CARRIER", "ORIGIN", "DEST",
    "DEP_DELAY", "ARR_DELAY", "FL_DATE"
]

# ---------- Schema detection & normalization ----------
def read_header(csv_path: str) -> List[str]:
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        if "," not in first:
            df = pd.read_csv(csv_path, nrows=0)
            return [c for c in df.columns]
        return [c.strip() for c in first.strip().split(",")]

def detect_schema(csv_path: str) -> str:
    cols = {c.upper() for c in read_header(csv_path)}
    if {"AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DEPARTURE_DELAY", "ARRIVAL_DELAY"} <= cols:
        return "KAGGLE"
    if {"OP_CARRIER", "ORIGIN", "DEST", "DEP_DELAY", "ARR_DELAY"} <= cols:
        return "BTS"
    df = pd.read_csv(csv_path, nrows=0)
    up = {c.upper() for c in df.columns}
    if {"AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DEPARTURE_DELAY", "ARRIVAL_DELAY"} <= up:
        return "KAGGLE"
    if {"OP_CARRIER", "ORIGIN", "DEST", "DEP_DELAY", "ARR_DELAY"} <= up:
        return "BTS"
    raise ValueError("Could not detect schema (KAGGLE or BTS).")

def normalize_df(df: pd.DataFrame, schema: str) -> pd.DataFrame:
    d = df.copy()
    up = {c.upper(): c for c in d.columns}

    if schema == "KAGGLE":
        d["CARRIER"] = d[up["AIRLINE"]]
        d["ORIGIN"] = d[up["ORIGIN_AIRPORT"]]
        d["DEST"] = d[up["DESTINATION_AIRPORT"]]
        d["DEP_DELAY"] = pd.to_numeric(d[up["DEPARTURE_DELAY"]], errors="coerce")
        d["ARR_DELAY"] = pd.to_numeric(d[up["ARRIVAL_DELAY"]], errors="coerce")
        year_col = up.get("YEAR")
        month_col = up.get("MONTH")
        if year_col and month_col:
            y = pd.to_numeric(d[year_col], errors="coerce").astype("Int64")
            m = pd.to_numeric(d[month_col], errors="coerce").astype("Int64")
            d["MONTH_LABEL"] = y.astype(str) + "-" + m.astype(str).str.zfill(2)
        elif "FL_DATE" in up:
            d["MONTH_LABEL"] = pd.to_datetime(d[up["FL_DATE"]], errors="coerce").dt.strftime("%Y-%m")
        else:
            d["MONTH_LABEL"] = pd.NaT

    elif schema == "BTS":
        d["CARRIER"] = d[up["OP_CARRIER"]]
        d["ORIGIN"] = d[up["ORIGIN"]]
        d["DEST"] = d[up["DEST"]]
        d["DEP_DELAY"] = pd.to_numeric(d[up["DEP_DELAY"]], errors="coerce")
        d["ARR_DELAY"] = pd.to_numeric(d[up["ARR_DELAY"]], errors="coerce")
        if "FL_DATE" in up:
            d["MONTH_LABEL"] = pd.to_datetime(d[up["FL_DATE"]], errors="coerce").dt.strftime("%Y-%m")
        else:
            year_col = up.get("YEAR")
            month_col = up.get("MONTH")
            if year_col and month_col:
                y = pd.to_numeric(d[year_col], errors="coerce").astype("Int64")
                m = pd.to_numeric(d[month_col], errors="coerce").astype("Int64")
                d["MONTH_LABEL"] = y.astype(str) + "-" + m.astype(str).str.zfill(2)
            else:
                d["MONTH_LABEL"] = pd.NaT
    else:
        raise ValueError(f"Unknown schema: {schema}")

    return d[["CARRIER", "ORIGIN", "DEST", "DEP_DELAY", "ARR_DELAY", "MONTH_LABEL"]]

# ---------- (Optional) enrichment via .env lookups ----------
AIRLINE_NAMES: Dict[str, str] = {}
AIRPORT_NAMES: Dict[str, str] = {}

def _load_code_name_map(path: str) -> Dict[str, str]:
    try:
        df = pd.read_csv(path)
        cols = list(df.columns)
        if len(cols) < 2:
            return {}
        code_candidates = [c for c in cols if str(c).lower() in ("iata_code", "iata", "carrier_code", "code", "airline", "iata_code_active")]
        name_candidates = [c for c in cols if str(c).lower() in ("airline", "airline_name", "name", "airport", "airport_name")]
        code_col = code_candidates[0] if code_candidates else cols[0]
        name_col = name_candidates[0] if name_candidates else cols[1]
        mapping = dict(zip(df[code_col].astype(str), df[name_col].astype(str)))
        return {k.strip(): v.strip() for k, v in mapping.items() if isinstance(k, str)}
    except Exception:
        return {}

AIRLINES_CSV = os.getenv("AIRLINES_CSV")
AIRPORTS_CSV = os.getenv("AIRPORTS_CSV")
if AIRLINES_CSV and os.path.exists(AIRLINES_CSV):
    AIRLINE_NAMES = _load_code_name_map(AIRLINES_CSV)
if AIRPORTS_CSV and os.path.exists(AIRPORTS_CSV):
    AIRPORT_NAMES = _load_code_name_map(AIRPORTS_CSV)

def maybe_enrich_airline(rows: List[dict]) -> List[dict]:
    if not AIRLINE_NAMES:
        return rows
    for r in rows:
        if "carrier" in r:
            r["airline_name"] = AIRLINE_NAMES.get(str(r["carrier"]), r["carrier"])
    return rows

def maybe_enrich_origin(rows: List[dict]) -> List[dict]:
    if not AIRPORT_NAMES:
        return rows
    for r in rows:
        if "origin" in r:
            r["origin_name"] = AIRPORT_NAMES.get(str(r["origin"]), r["origin"])
    return rows

# ---------- Aggregations (in-memory) ----------
def avg_arrival_delay_by_airline_df(df: pd.DataFrame) -> List[dict]:
    g = df.groupby("CARRIER", dropna=False)["ARR_DELAY"].mean(numeric_only=True)
    out = (
        g.reset_index()
         .rename(columns={"CARRIER": "carrier", "ARR_DELAY": "avg_arrival_delay"})
         .sort_values("avg_arrival_delay")
    )
    return maybe_enrich_airline(out.to_dict(orient="records"))

def flights_by_origin_df(df: pd.DataFrame) -> List[dict]:
    g = df.groupby("ORIGIN", dropna=False).size()
    out = (
        g.reset_index(name="flights")
         .rename(columns={"ORIGIN": "origin"})
         .sort_values("flights", ascending=False)
    )
    return maybe_enrich_origin(out.to_dict(orient="records"))

def avg_dep_delay_by_month_df(df: pd.DataFrame) -> List[dict]:
    g = df.groupby("MONTH_LABEL", dropna=False)["DEP_DELAY"].mean(numeric_only=True)
    out = (
        g.reset_index()
         .rename(columns={"MONTH_LABEL": "month", "DEP_DELAY": "avg_departure_delay"})
         .sort_values("month")
    )
    return out.to_dict(orient="records")

# ---------- Aggregations (chunked) ----------
def chunk_reader(csv_path: str, schema: str, chunksize: int) -> Iterable[pd.DataFrame]:
    usecols = REQUIRED_COLS_KAGGLE if schema == "KAGGLE" else REQUIRED_COLS_BTS
    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=str,
        chunksize=chunksize,
        low_memory=False
    ):
        yield normalize_df(chunk, schema)

def avg_arrival_delay_by_airline_chunked(csv_path: str, schema: str, chunksize: int) -> List[dict]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for c in chunk_reader(csv_path, schema, chunksize):
        grp = c.groupby("CARRIER", dropna=False)["ARR_DELAY"].agg(["sum", "count"])
        for carrier, row in grp.iterrows():
            sums[carrier] = sums.get(carrier, 0.0) + float(row["sum"])
            counts[carrier] = counts.get(carrier, 0) + int(row["count"])
    rows = [{"carrier": k, "avg_arrival_delay": (sums[k] / (counts[k] or 1))} for k in sums]
    rows.sort(key=lambda r: r["avg_arrival_delay"])
    return maybe_enrich_airline(rows)

def flights_by_origin_chunked(csv_path: str, schema: str, chunksize: int) -> List[dict]:
    counts: Dict[str, int] = {}
    for c in chunk_reader(csv_path, schema, chunksize):
        grp = c.groupby("ORIGIN", dropna=False).size()
        for origin, n in grp.items():
            counts[origin] = counts.get(origin, 0) + int(n)
    rows = [{"origin": k, "flights": v} for k, v in counts.items()]
    rows.sort(key=lambda r: r["flights"], reverse=True)
    return maybe_enrich_origin(rows)

def avg_dep_delay_by_month_chunked(csv_path: str, schema: str, chunksize: int) -> List[dict]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for c in chunk_reader(csv_path, schema, chunksize):
        grp = c.groupby("MONTH_LABEL", dropna=False)["DEP_DELAY"].agg(["sum", "count"])
        for month, row in grp.iterrows():
            sums[month] = sums.get(month, 0.0) + float(row["sum"])
            counts[month] = counts.get(month, 0) + int(row["count"])
    rows = [{"month": m, "avg_departure_delay": (sums[m] / (counts[m] or 1))} for m in sums]
    rows.sort(key=lambda r: (r["month"] is None, r["month"]))
    return rows

# ---------- Query registry ----------
AGGS = {
    "avg_arr_delay_by_airline": (avg_arrival_delay_by_airline_df,  avg_arrival_delay_by_airline_chunked),
    "flights_by_origin":        (flights_by_origin_df,             flights_by_origin_chunked),
    "avg_dep_delay_by_month":   (avg_dep_delay_by_month_df,        avg_dep_delay_by_month_chunked),
}

# ---------- Execution ----------
def run_query(query_name: str, csv_path: str, schema: str, chunksize: int = 0, enrich: bool = True) -> List[dict]:
    if query_name not in AGGS:
        raise ValueError(f"Unknown query: {query_name}")

    params = {
        "csv": str(Path(csv_path).resolve()),
        "schema": schema,
        "chunksize": int(chunksize),
        "v": 4,
        "enrich": bool(enrich),
    }
    cache_key = key_for(query_name, params)

    cached = get_json(cache_key)
    if cached is not None:
        logging.info("CACHE_HIT key=%s", cache_key)
        incr("metrics:hits")
        return cached

    logging.info("CACHE_MISS key=%s", cache_key)
    incr("metrics:misses")

    if chunksize and chunksize > 0:
        compute_fn = AGGS[query_name][1]
        result = compute_fn(csv_path, schema, chunksize)  # type: ignore
    else:
        usecols = REQUIRED_COLS_KAGGLE if schema == "KAGGLE" else REQUIRED_COLS_BTS
        df = pd.read_csv(csv_path, usecols=usecols, dtype=str, low_memory=False)
        df = normalize_df(df, schema)
        compute_fn = AGGS[query_name][0]
        result = compute_fn(df)  # type: ignore

    set_json(cache_key, result, ttl=DEFAULT_TTL)
    return result

# ---------- CLI ----------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run cached CSV analytics")
    p.add_argument("--query", default="avg_arr_delay_by_airline", choices=list(AGGS.keys()))
    p.add_argument("--csv", default=CSV_PATH_DEFAULT, help=f"Path to CSV (default: {CSV_PATH_DEFAULT})")
    p.add_argument("--clear-cache", action="store_true", help="Clear cache entries with prefix 'agg:' and exit")
    p.add_argument("--show-cache", action="store_true", help="List cached keys (and value snippets) and exit")
    p.add_argument("--full", action="store_true", help="With --show-cache, pretty-print full JSON values")
    p.add_argument("--limit", type=int, default=PRINT_LIMIT_DEFAULT, help=f"Limit rows printed (default: {PRINT_LIMIT_DEFAULT})")
    return p

# ---------- Main ----------
if __name__ == "__main__":
    p = build_arg_parser()
    args = p.parse_args()

    if args.show_cache:
        entries = list_cache()
        if not entries:
            print("No cache entries found.")
            sys.exit(0)

        print(f"Found {len(entries)} cached keys:\n")
        if args.full:
            for k, _ in entries:
                val = get_json(k)
                print(k)
                if val is None:
                    print("  (nil)\n")
                else:
                    print(json.dumps(val, indent=2)[:100000], "\n")
        else:
            for k, v in entries:
                print(f"{k} -> {v}")
        sys.exit(0)

    if args.clear_cache:
        deleted = clear_prefix("agg:")
        logging.info("Cache cleared (%s keys deleted)", deleted)
        sys.exit(0)

    # Friendly CSV existence check
    if not os.path.exists(args.csv):
        print(
            f"CSV not found: {args.csv}\n"
            "Tip: pass --csv PATH or set CSV_PATH in your .env.\n"
            "Expected headers (either schema):\n"
            "  Kaggle: AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, DEPARTURE_DELAY, ARRIVAL_DELAY, YEAR, MONTH\n"
            "  BTS:    OP_CARRIER, ORIGIN, DEST, DEP_DELAY, ARR_DELAY, FL_DATE"
        )
        sys.exit(1)

    schema = detect_schema(args.csv)
    logging.info("Detected schema: %s", schema)

    chunksize = 0
    if CHUNK_SIZE_ENV:
        try:
            chunksize = int(CHUNK_SIZE_ENV)
            logging.info("Chunked mode enabled: CHUNKSIZE=%s", chunksize)
        except ValueError:
            logging.warning("Invalid CHUNKSIZE env value '%s' (ignored)", CHUNK_SIZE_ENV)

    result = run_query(
        args.query,
        args.csv,
        schema,
        chunksize=chunksize,
        enrich=True,
    )

    to_print = result if args.limit is None else result[: args.limit]
    print(json.dumps(to_print, indent=2))
