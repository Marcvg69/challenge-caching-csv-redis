#!/usr/bin/env python3
"""
run_benchmark.py — time MISS vs HIT by invoking the CLI (python -m app.main)

This avoids brittle imports and guarantees we exercise the same code paths
your users run from the command line.
"""

import argparse
import subprocess
import time
from pathlib import Path

import pandas as pd

START_TAG = "<!-- BENCHMARK:START -->"
END_TAG = "<!-- BENCHMARK:END -->"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command, capture output, raise on error."""
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def bench_query(query: str) -> dict:
    """Measure MISS then HIT by calling the CLI twice."""
    # Clear cache via CLI
    _run(["python", "-m", "app.main", "--clear-cache"])

    # MISS
    t0 = time.perf_counter()
    _run(["python", "-m", "app.main", "--query", query, "--limit", "5"])
    miss = time.perf_counter() - t0

    # HIT
    t1 = time.perf_counter()
    _run(["python", "-m", "app.main", "--query", query, "--limit", "5"])
    hit = time.perf_counter() - t1

    return {"query": query, "miss_sec": round(miss, 3), "hit_sec": round(hit, 3)}


def update_readme(block: str, readme_path: Path = Path("README.md")) -> bool:
    """
    Replace or append the benchmark block in README.md.
    Returns True if README was updated.
    """
    if not readme_path.exists():
        return False

    text = readme_path.read_text(encoding="utf-8")
    new_block = f"{START_TAG}\n\n{block}\n{END_TAG}"

    if START_TAG in text and END_TAG in text:
        start = text.index(START_TAG)
        end = text.index(END_TAG) + len(END_TAG)
        updated = text[:start] + new_block + text[end:]
    else:
        appendix = "\n\n## Benchmark\n\n" + new_block + "\n"
        updated = text + appendix

    readme_path.write_text(updated, encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-readme", action="store_true", help="Update README.md with results")
    args = parser.parse_args()

    queries = [
        "avg_arr_delay_by_airline",
        "flights_by_origin",
        "avg_dep_delay_by_month",
    ]

    rows = []
    for q in queries:
        rows.append(bench_query(q))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if args.update_readme:
        block = df.to_markdown(index=False)
        if update_readme(block):
            print("✅ README.md updated with benchmark results")
        else:
            print("⚠️ README.md not found")


if __name__ == "__main__":
    main()
