"""
Large CSV Caching Demo (Kaggle-compatible, chunk-aware, with optional enrichment)

What this program does
----------------------
- Reads a large CSV (path from --csv or .env CSV_PATH). Supports Kaggle 'usdot/flight-delays'.
- Computes aggregations (groupbys).
- Caches each aggregation result in Redis with a TTL.
- Logs CACHE_HIT (fast) vs CACHE_MISS (compute+cache) with timings.
- Handles very large files by streaming in chunks (CHUNKSIZE in .env).
- Optionally enriches results with airline and airport names if AIRLINES_CSV / AIRPORTS_CSV are set.

Why chunking
------------
If your CSV is hundreds of MB or more, reading it all into RAM can be slow or impossible.
With chunking we aggregate incrementally and keep memory usage almost constant.

Supported schemas
-----------------
1) Kaggle 'usdot/flight-delays' (columns like: AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT,
   DEPARTURE_DELAY, ARRIVAL_DELAY, YEAR, MONTH, DAY, ...)
2) BTS-style files with columns like: OP_CARRIER, ORIGIN, DEST, DEP_DELAY, ARR_DELAY, FL_DATE

We normalize both to a common set inside the code:
  CARRIER, ORIGIN, DEST, DEP_DELAY, ARR_DELAY, MONTH_LABEL (YYYY-MM string)

Run examples
------------
    python app/main.py --query avg_arr_delay_by_airline
    python app/main.py --query flights_by_origin
    python app/main.py --query avg_dep_delay_by_month
    python app/main.py --clear-cache
"""
import os, time, argparse, logging, json
import pandas as pd
from dotenv import load_dotenv
from cache import key_for, get_json, set_json, incr, clear_prefix

# ----------------- Setup & config -----------------
load_dotenv()

CSV_PATH      = os.getenv("CSV_PATH", "./data/flights.csv")
AIRLINES_CSV  = os.getenv("AIRLINES_CSV")  # optional
AIRPORTS_CSV  = os.getenv("AIRPORTS_CSV")  # optional
CHUNKSIZE     = int(os.getenv("CHUNKSIZE", "0"))  # 0 = in-memory, >0 = chunked streaming

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------- Schema detection -----------------
def detect_schema(csv_path: str) -> str:
    """
    Peek at header to decide which column names to expect.
    Returns 'KAGGLE' or 'BTS'.
    """
    cols = pd.read_csv(csv_path, nrows=0).columns
    if "AIRLINE" in cols and "ORIGIN_AIRPORT" in cols:
        return "KAGGLE"
    if "OP_CARRIER" in cols and "ORIGIN" in cols:
        return "BTS"
    raise SystemExit("Unsupported CSV schema. Please provide Kaggle 'flights.csv' or BTS-style file.")

def usecols_for(schema: str):
    """Columns to read from the CSV for each schema (minimal set to save memory)."""
    if schema == "KAGGLE":
        # We build month labels from YEAR+MONTH; DAY not required for these aggs.
        return ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
                "DEPARTURE_DELAY", "ARRIVAL_DELAY", "YEAR", "MONTH"]
    # BTS
    return ["OP_CARRIER", "ORIGIN", "DEST", "DEP_DELAY", "ARR_DELAY", "FL_DATE"]

def normalize_columns(df: pd.DataFrame, schema: str) -> pd.DataFrame:
    """
    Convert the DataFrame to a canonical schema:
      CARRIER, ORIGIN, DEST, DEP_DELAY, ARR_DELAY, MONTH_LABEL
    - For KAGGLE: MONTH_LABEL = f"{YEAR:04d}-{MONTH:02d}"
    - For BTS:    MONTH_LABEL from FL_DATE.to_period('M')
    """
    if schema == "KAGGLE":
        df = df.rename(columns={
            "AIRLINE": "CARRIER",
            "ORIGIN_AIRPORT": "ORIGIN",
            "DESTINATION_AIRPORT": "DEST",
            "DEPARTURE_DELAY": "DEP_DELAY",
            "ARRIVAL_DELAY": "ARR_DELAY",
        })
        # Build YYYY-MM label from YEAR/MONTH (both are ints in Kaggle dataset)
        df["MONTH_LABEL"] = (df["YEAR"].astype(int)).astype(str) + "-" + df["MONTH"].astype(int).map("{:02d}".format)
        return df[["CARRIER", "ORIGIN", "DEST", "DEP_DELAY", "ARR_DELAY", "MONTH_LABEL"]]
    else:
        # BTS-style: parse month from FL_DATE
        df = df.rename(columns={
            "OP_CARRIER": "CARRIER",
            "DEP_DELAY":  "DEP_DELAY",
            "ARR_DELAY":  "ARR_DELAY",
        })
        # Ensure FL_DATE is a datetime to derive month label
        if not pd.api.types.is_datetime64_any_dtype(df.get("FL_DATE")):
            df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
        df["MONTH_LABEL"] = df["FL_DATE"].dt.to_period("M").astype(str)
        return df[["CARRIER", "ORIGIN", "DEST", "DEP_DELAY", "ARR_DELAY", "MONTH_LABEL"]]

# ----------------- Optional enrichment dictionaries -----------------
def load_airline_dict():
    """
    Returns {carrier_code: full_name} if AIRLINES_CSV is set and readable.
    Kaggle airlines.csv schema: IATA_CODE,AIRLINE
    """
    if not AIRLINES_CSV or not os.path.exists(AIRLINES_CSV):
        return None
    try:
        a = pd.read_csv(AIRLINES_CSV)
        if "IATA_CODE" in a.columns and "AIRLINE" in a.columns:
            return dict(zip(a["IATA_CODE"], a["AIRLINE"]))
    except Exception as e:
        logging.warning("Could not load AIRLINES_CSV: %s", e)
    return None

def load_airport_dict():
    """
    Returns {iata_code: airport_name} if AIRPORTS_CSV is set and readable.
    Kaggle airports.csv schema: IATA_CODE,AIRPORT,...
    """
    if not AIRPORTS_CSV or not os.path.exists(AIRPORTS_CSV):
        return None
    try:
        a = pd.read_csv(AIRPORTS_CSV)
        if "IATA_CODE" in a.columns and "AIRPORT" in a.columns:
            return dict(zip(a["IATA_CODE"], a["AIRPORT"]))
    except Exception as e:
        logging.warning("Could not load AIRPORTS_CSV: %s", e)
    return None

AIRLINE_NAME = load_airline_dict()  # small; safe to keep in memory
AIRPORT_NAME = load_airport_dict()  # small; safe to keep in memory

# ----------------- IO helpers -----------------
def load_df(csv_path: str, schema: str) -> pd.DataFrame:
    """Load the whole file (for smaller CSVs) and normalize columns."""
    df = pd.read_csv(
        csv_path,
        usecols=usecols_for(schema),
        low_memory=False,
    )
    return normalize_columns(df, schema)

def iter_chunks(csv_path: str, schema: str, chunksize: int):
    """Yield normalized chunks for huge files."""
    for c in pd.read_csv(
        csv_path,
        usecols=usecols_for(schema),
        low_memory=False,
        chunksize=chunksize,
    ):
        yield normalize_columns(c, schema)

# ----------------- Aggregations (operate on normalized columns) -----------------
def avg_arrival_delay_by_airline_df(df: pd.DataFrame):
    out = df.groupby("CARRIER")["ARR_DELAY"].mean().sort_values().reset_index()
    out.columns = ["carrier", "avg_arrival_delay"]
    # Optional: add airline names if mapping is available
    if AIRLINE_NAME:
        out["airline_name"] = out["carrier"].map(AIRLINE_NAME)
    return out

def flights_by_origin_df(df: pd.DataFrame):
    out = df.groupby("ORIGIN").size().sort_values(ascending=False).reset_index(name="flights")
    out.columns = ["origin", "flights"]
    if AIRPORT_NAME:
        out["origin_name"] = out["origin"].map(AIRPORT_NAME)
    return out

def avg_dep_delay_by_month_df(df: pd.DataFrame):
    out = df.groupby("MONTH_LABEL")["DEP_DELAY"].mean().reset_index()
    out.columns = ["month", "avg_departure_delay"]
    return out

# ----------------- Chunked incremental aggregations -----------------
def avg_arrival_delay_by_airline_chunked(chunks):
    sums, counts = {}, {}
    for c in chunks:
        g = c.groupby("CARRIER")["ARR_DELAY"].agg(["sum", "count"])
        for carrier, row in g.iterrows():
            sums[carrier]   = sums.get(carrier, 0.0) + float(row["sum"])
            counts[carrier] = counts.get(carrier, 0)   + int(row["count"])
    rows = [{"carrier": k, "avg_arrival_delay": (sums[k] / counts[k])} for k in sums if counts[k] > 0]
    out = pd.DataFrame(rows).sort_values("avg_arrival_delay").reset_index(drop=True)
    if AIRLINE_NAME:
        out["airline_name"] = out["carrier"].map(AIRLINE_NAME)
    return out

def flights_by_origin_chunked(chunks):
    counts = {}
    for c in chunks:
        g = c.groupby("ORIGIN").size()
        for k, v in g.items():
            counts[k] = counts.get(k, 0) + int(v)
    out = pd.DataFrame([{"origin": k, "flights": v} for k, v in counts.items()]) \
            .sort_values("flights", ascending=False).reset_index(drop=True)
    if AIRPORT_NAME:
        out["origin_name"] = out["origin"].map(AIRPORT_NAME)
    return out

def avg_dep_delay_by_month_chunked(chunks):
    sums, counts = {}, {}
    for c in chunks:
        g = c.groupby("MONTH_LABEL")["DEP_DELAY"].agg(["sum", "count"])
        for m, row in g.iterrows():
            sums[m]   = sums.get(m, 0.0) + float(row["sum"])
            counts[m] = counts.get(m, 0)   + int(row["count"])
    rows = [{"month": k, "avg_departure_delay": (sums[k] / counts[k])} for k in sums if counts[k] > 0]
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)

# Map query name -> (in-memory fn, chunked fn)
AGGS = {
    "avg_arr_delay_by_airline": (avg_arrival_delay_by_airline_df,  avg_arrival_delay_by_airline_chunked),
    "flights_by_origin":        (flights_by_origin_df,             flights_by_origin_chunked),
    "avg_dep_delay_by_month":   (avg_dep_delay_by_month_df,        avg_dep_delay_by_month_chunked),
}

# ----------------- Runner with caching -----------------
def run_query(query_name: str, csv_path: str):
    if query_name not in AGGS:
        raise SystemExit(f"Unknown query '{query_name}'. Choices: {list(AGGS)}")

    schema = detect_schema(csv_path)

    # Cache key: depends on CSV path, query, logic version, chunksize AND schema
    params = {"csv": os.path.abspath(csv_path), "v": 3, "chunksize": CHUNKSIZE, "schema": schema}
    cache_key = key_for(query_name, params)

    # Try cache first
    t0 = time.perf_counter()
    cached = get_json(cache_key)
    if cached is not None:
        dt = (time.perf_counter() - t0) * 1000
        logging.info("CACHE_HIT key=%s (%.1f ms)", cache_key, dt)
        incr("metrics:hits")
        return cached

    # Compute
    inmem_fn, chunked_fn = AGGS[query_name]
    if CHUNKSIZE > 0:
        logging.info("Chunked mode enabled (CHUNKSIZE=%s, schema=%s)", CHUNKSIZE, schema)
        chunks = iter_chunks(csv_path, schema, CHUNKSIZE)
        result_df = chunked_fn(chunks)
    else:
        df = load_df(csv_path, schema)
        result_df = inmem_fn(df)

    result = result_df.to_dict(orient="records")
    set_json(cache_key, result)
    logging.info("CACHE_MISS computed; cached key=%s", cache_key)
    incr("metrics:misses")
    return result

def main():
    p = argparse.ArgumentParser(description="Large CSV caching demo with Redis.")
    p.add_argument("--query", default="avg_arr_delay_by_airline", choices=list(AGGS.keys()))
    p.add_argument("--csv", default=CSV_PATH, help="Path to CSV (overrides .env CSV_PATH).")
    p.add_argument("--clear-cache", action="store_true", help="Clear cached aggregations and exit.")
    args = p.parse_args()

    if args.clear_cache:
        clear_prefix("agg:")
        logging.info("Cache cleared.")
        return

    logging.info("Running query=%s on csv=%s", args.query, args.csv)
    t0 = time.perf_counter()
    result = run_query(args.query, args.csv)
    total_ms = (time.perf_counter() - t0) * 1000
    logging.info("Done in %.1f ms", total_ms)

    # Limit terminal noise: show first 10 rows
    print(json.dumps(result[:10], indent=2))

if __name__ == "__main__":
    main()
