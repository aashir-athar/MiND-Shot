#!/usr/bin/env python3
"""
One-shot fetch of deep 4h history from Binance's public archive (data.binance.vision)
into ``data/<SYMBOL>_4h.csv`` — same schema as the committed test fixtures.

Run from the repo root whenever you want to refresh the committed history:

    python tools/fetch_history.py

The archive is a plain CDN (not the geo-blocked API). Timestamps are normalised
to milliseconds (Binance switched the archive to microseconds in 2025). Months
already present are skipped, so re-runs only append the new tail. Pure stdlib.
"""
from __future__ import annotations

import csv
import io
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
START_YEAR, START_MONTH = 2020, 1
BASE = "https://data.binance.vision/data/spot/monthly/klines/{sym}/4h/{sym}-4h-{y}-{m:02d}.zip"


def norm_ms(ts: int) -> int:
    """Archive timestamps: seconds, milliseconds, or microseconds → milliseconds."""
    if ts > 10**14:
        return ts // 1000
    if ts > 10**11:
        return ts
    return ts * 1000


def months(until: datetime):
    y, m = START_YEAR, START_MONTH
    while (y, m) < (until.year, until.month):        # last COMPLETE month only
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fetch_month(sym: str, y: int, m: int):
    url = BASE.format(sym=sym, y=y, m=m)
    blob = None
    for attempt in range(4):                          # CDN reads occasionally stall
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                blob = r.read()
            break
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return []                             # symbol not listed yet that month
            if attempt == 3:
                raise
        except (TimeoutError, OSError):
            if attempt == 3:
                raise
    if blob is None:
        return []
    rows = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                parts = line.strip().split(",")
                if not parts or not parts[0] or not parts[0][0].isdigit():
                    continue                          # header / blank
                # open_time, open, high, low, close, volume, close_time, ...
                rows.append((norm_ms(int(parts[0])), parts[1], parts[2], parts[3],
                             parts[4], parts[5], norm_ms(int(parts[6]))))
    return rows


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    now = datetime.now(tz=timezone.utc)
    for sym in SYMBOLS:
        out = OUT_DIR / f"{sym}_4h.csv"
        existing = {}
        if out.exists():
            with open(out, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    existing[int(row["open_time"])] = (
                        row["open_time"], row["open"], row["high"], row["low"],
                        row["close"], row["volume"], row["close_time"])
        have_until = max(existing) if existing else 0
        added = 0
        for y, m in months(now):
            month_start_ms = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000)
            if month_start_ms + 27 * 86400_000 < have_until:
                continue                              # month already covered
            for r in fetch_month(sym, y, m):
                if r[0] not in existing:
                    existing[r[0]] = (str(r[0]), r[1], r[2], r[3], r[4], r[5], str(r[6]))
                    added += 1
            print(f"  {sym} {y}-{m:02d} ok", file=sys.stderr)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["open_time", "open", "high", "low", "close", "volume", "close_time"])
            for t in sorted(existing):
                w.writerow(existing[t])
        span = (max(existing) - min(existing)) / 86400_000 if existing else 0
        print(f"{sym}: {len(existing)} bars over {span:.0f} days (+{added} new) -> {out}")


if __name__ == "__main__":
    main()
