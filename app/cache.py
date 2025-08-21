"""
Cache utilities for Redis:
- Connect via env (.env)
- Build stable keys from (query_name, params)
- Get/Set JSON values with TTL
- Increment simple counters for HIT/MISS metrics
"""
import os, json, logging
from hashlib import sha1
from dotenv import load_dotenv
import redis

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))
TTL_SEC    = int(os.getenv("CACHE_TTL_SECONDS", "60"))

_pool = None

def get_client():
    """Return a Redis client using a shared connection pool."""
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
        )
    return redis.Redis(connection_pool=_pool)

def key_for(query_name: str, params: dict) -> str:
    """Stable cache key from query name + params (order-insensitive)."""
    payload = json.dumps({"q": query_name, "p": params}, sort_keys=True, default=str)
    return f"agg:{query_name}:{sha1(payload.encode()).hexdigest()[:16]}"

def get_json(key: str):
    """Get JSON from Redis (returns Python object or None)."""
    r = get_client()
    raw = r.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        logging.warning("Cache decode failed for key=%s", key)
        return None

def set_json(key: str, value, ttl: int = TTL_SEC):
    """Set JSON with TTL (seconds)."""
    r = get_client()
    r.set(key, json.dumps(value), ex=ttl)

def incr(metric_key: str):
    """Increment a numeric counter (e.g., metrics:hits / metrics:misses)."""
    r = get_client()
    r.incr(metric_key)

def clear_prefix(prefix="agg:"):
    """Delete all keys that start with 'prefix' (careful in shared Redis!)."""
    r = get_client()
    for k in r.scan_iter(match=f"{prefix}*"):
        r.delete(k)

def list_cache(prefix="agg:"):
"""List keys and values currently cached in Redis"""
    keys = r.keys(f"{prefix}*")
    result = []
    for k in keys:
        val = r.get(k)
        if val:
            snippet = val[:80] + ("..." if len(val) > 80 else "")
            result.append((k, snippet))
        else:
            result.append((k, None))
    return result
