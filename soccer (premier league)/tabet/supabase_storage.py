"""Minimal client for private Supabase Storage buckets (stdlib only, so the logger needs no new package).

Uses SUPABASE_URL and SUPABASE_SECRET_KEY from the environment (GitHub secrets on Actions, the local
backend/.env otherwise). Objects are private: only the secret key can read or write them.
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def configured() -> bool:
    return bool(os.environ.get("SUPABASE_URL", "").strip() and os.environ.get("SUPABASE_SECRET_KEY", "").strip())


def _request(method: str, path: str, data: bytes | None = None, content_type: str | None = None, extra=None) -> bytes:
    url = os.environ["SUPABASE_URL"].strip().rstrip("/")
    key = os.environ["SUPABASE_SECRET_KEY"].strip()
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    if content_type:
        headers["Content-Type"] = content_type
    headers.update(extra or {})
    req = urllib.request.Request(url + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60, context=_ctx()) as r:
            return r.read()
    except urllib.error.HTTPError as e:   # never echo the key; the path and status are enough
        raise RuntimeError(f"Supabase storage {method} {path.split('?')[0]}: HTTP {e.code} {e.read()[:200]!r}") from None


def put(bucket: str, path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload (or overwrite) one object; returns 'supabase:<bucket>/<path>' for logs."""
    _request("POST", f"/storage/v1/object/{bucket}/{urllib.parse.quote(path)}", data, content_type, {"x-upsert": "true"})
    return f"supabase:{bucket}/{path}"


def get(bucket: str, path: str) -> bytes:
    return _request("GET", f"/storage/v1/object/{bucket}/{urllib.parse.quote(path)}")


def ensure_bucket(bucket: str) -> None:
    """Create a private bucket if it does not exist."""
    try:
        _request("POST", "/storage/v1/bucket", json.dumps({"id": bucket, "name": bucket, "public": False}).encode(),
                 "application/json")
    except RuntimeError as e:
        if "409" not in str(e) and "already exists" not in str(e).lower() and "Duplicate" not in str(e):
            raise
