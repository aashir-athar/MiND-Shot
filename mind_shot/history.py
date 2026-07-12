"""
Deep committed history — loader for ``data/<SYMBOL>_4h.csv``.

``tools/fetch_history.py`` snapshots years of Binance 4h candles into ``data/``
(same CSV schema as the test fixtures). This module merges that deep history
with the committed fixtures, deduplicated by timestamp, oldest→newest — the
single source both the weekly trainer and the ML evaluation replay train on.

The playbook fixtures in ``tests/fixtures/`` are untouched and remain the sole
input to the validation backtest; deep history adds bars OUTSIDE the fixture
window (years before it, and whatever tail has accrued since the snapshot) —
where both sources have a bar for the same timestamp, the fixture bar wins.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List

from .backtest import _load_fixture
from .models import Candle

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}


def has_deep_history(asset: str) -> bool:
    return os.path.isfile(os.path.join(DATA_DIR, f"{_SYMBOL.get(asset, '')}_4h.csv"))


def load_history(asset: str) -> List[Candle]:
    """Deep history ∪ fixtures for ``asset``, deduped by open time (seconds)."""
    merged: Dict[int, Candle] = {c[0]: c for c in _load_fixture(asset)}
    path = os.path.join(DATA_DIR, f"{_SYMBOL[asset]}_4h.csv")
    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    c = (int(row["open_time"]) // 1000,
                         float(row["open"]), float(row["high"]), float(row["low"]),
                         float(row["close"]), float(row["volume"]))
                except (KeyError, TypeError, ValueError):
                    continue
                merged.setdefault(c[0], c)
    return [merged[t] for t in sorted(merged)]


__all__ = ["load_history", "has_deep_history", "DATA_DIR"]
