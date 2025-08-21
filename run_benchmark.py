#!/usr/bin/env python3
"""
run_benchmark.py
- Benchmarks dictionary HIT vs MISS lookup times.
- Prints a Markdown table (no external deps).
- Optionally updates README.md between <!-- BENCHMARK:START --> and <!-- BENCHMARK:END -->.

Usage:
  python run_benchmark.py --iters 200000 --miss-ratio 0.4 --update-readme
"""

import argparse
import random
import statistics
import time
from pathlib import Path

# ---------- Core benchmark ----------

def benchmark(n: int = 100_000, miss_ratio: float = 0.5):
    """
    Compare HIT vs MISS lookups in a Python dict.

    Args:
        n: number of total lookups.
        miss_ratio: fraction (0–1) of lookups that should MISS.

    Returns:
        dict with average, median and stdev (ns) for HIT and MISS.
    """
    # Simulated cache: keys "0".."999"
    data = {str(i): i for i in range(1000)}
    hit_times, miss_times = [], []

    # Local bindings for speed
    rnd = random.random
    randint = random.randint
    get = data.get
    perf = time.perf_counter_ns

    for _ in range(n):
        if rnd() > miss_ratio:
            # HIT: choose an existing key
            key = str(randint(0, 999))
            t0 = perf(); _ = get(key); dt = perf() - t0
            hit_times.append(dt)
        else:
            # MISS: choose a non-existing key
            key = str(randint(1000, 2000))
            t0 = perf(); _ = get(key); dt = perf() - t0
            miss_times.append(dt)

    def stats(xs):
        # Guard against empty list if miss_ratio is 0 or 1
        if not xs:
            return {"avg": float("nan"), "med": float("nan"), "std": float("nan"), "n": 0}
        return {
            "avg": statistics.fmean(xs),
            "med": statistics.median(xs),
            "std": statistics.pstdev(xs),
            "n": len(xs),
        }

    return {"HIT": stats(hit_times), "MISS": stats(miss_times)}

# ---------- Markdown rendering (no tabulate dependency) ----------

def md_table(results: dict, iters: int, miss_ratio: float) -> str:
    """
    Build a GitHub-flavoured Markdown table string.
    """
    header = (
        f"**Dictionary lookup benchmark** — iters={iters:,}, miss_ratio={miss_ratio:.2f}\n\n"
        "| Operation | n | Avg (ns) | Median (ns) | Std (ns) |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    def row(name):
        s = results[name]
        fmt = lambda x: "nan" if x != x else f"{x:.1f}"  # simple NaN-safe
        return f"| {name} | {s['n']:,} | {fmt(s['avg'])} | {fmt(s['med'])} | {fmt(s['std'])} |"
    return header + "\n".join([row("HIT"), row("MISS")]) + "\n"

# ---------- README updater ----------

START_TAG = "<!-- BENCHMARK:START -->"
END_TAG   = "<!-- BENCHMARK:END -->"

def update_readme(block: str, readme_path: Path = Path("README.md")) -> bool:
    """
    Replace or append the benchmark block in README.md.

    Returns:
        True if README was updated, False if README not found.
    """
    if not readme_path.exists():
        return False

    text = readme_path.read_text(encoding="utf-8")

    new_block = f"{START_TAG}\n\n{block}\n{END_TAG}"

    if START_TAG in text and END_TAG in text:
        # Replace existing block
        start = text.index(START_TAG)
        end = text.index(END_TAG) + len(END_TAG)
        updated = text[:start] + new_block + text[end:]
    else:
        # Append a new block at the end with a header
        appendix = "\n\n## Benchmark\n\n" + new_block + "\n"
        updated = text + appendix

    readme_path.write_text(updated, encoding="utf-8")
    return True

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark dict HIT vs MISS and output Markdown.")
    p.add_argument("--iters", type=int, default=100_000, help="Total number of lookups (default: 100000)")
    p.add_argument("--miss-ratio", type=float, default=0.5, help="Fraction of MISS lookups between 0 and 1 (default: 0.5)")
    p.add_argument("--update-readme", action="store_true", help="Update README.md between BENCHMARK tags")
    return p.parse_args()

def main():
    args = parse_args()
    results = benchmark(n=args.iters, miss_ratio=args.miss_ratio)
    table = md_table(results, iters=args.iters, miss_ratio=args.miss_ratio)

    # Always print to console for copy-paste
    print(table)

    if args.update_readme:
        ok = update_readme(table)
        if ok:
            print("README.md updated ✅")
        else:
            print("README.md not found — skipped update.")

if __name__ == "__main__":
    main()
