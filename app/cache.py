#!/usr/bin/env python3
"""
cache.py — Redis helpers for cached CSV analytics
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import List, Tuple

import redis

# ---------------------------------------------------------------------------
# Minimal .env loader (no external deps). We load this BEFORE reading env vars.
# Looks for .env at repo root, then app/.env.
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
        break  # stop at first .env found

load_env(override=False)

# ---------------------------------------------------------------------------
# Connection (override via .env or process env)
#   REDIS_HOST=localhost
#   REDIS_PORT=6380   # if your compose maps 6380->6379
# ---------------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None

# Single global Redis client (decode_responses=True -> str in/out)
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True,
)

# ---------------------------------------------------------------------------
# Key builder — stable key from query name + hashed params
# ---------------------------------------------------------------------------
def key_for(query_name: str, params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"agg:{query_name}:{h}"

# ---------------------------------------------------------------------------
# JSON get/set
# ---------------------------------------------------------------------------
def get_json(key: str):
    raw = r.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        # If value isn't valid JSON, just return raw string
        return raw

def set_json(key: str, value, ttl: int = 3600) -> None:
    r.setex(key, ttl, json.dumps(value))

# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def incr(metric_key: str) -> None:
    r.incr(metric_key)

# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------
def clear_prefix(prefix: str = "agg:") -> int:
    """Delete all keys with the given prefix. Returns number of deleted keys."""
    count = 0
    pattern = f"{prefix}*"
    for k in r.scan_iter(match=pattern, count=500):
        r.delete(k)
        count += 1
    return count

def list_cache(prefix: str = "agg:") -> List[Tuple[str, str | None]]:
    """
    Return a list of (key, snippet) for keys under prefix.
    Snippet shows the first 80 chars of the cached value (if any).
    """
    out: List[Tuple[str, str | None]] = []
    pattern = f"{prefix}*"
    for k in r.scan_iter(match=pattern, count=500):
        val = r.get(k)
        if val is None:
            out.append((k, None))
        else:
            snippet = val[:80] + ("..." if len(val) > 80 else "")
            out.append((k, snippet))
    return out
