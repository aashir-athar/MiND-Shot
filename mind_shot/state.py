"""
Persistent state — git-committed JSON, loaded once per poll and saved at the end.

Layout (unchanged from v1 so existing dashboards / the Electron host keep working):

    {
      "<strategy_id>": {"active_trade": {...}|null, "last_signal_bar": int, "ml": {...}},
      ...
      "__global": {daily_r, heatmap, streaks, journal, risk controls, sl_cooldowns, ...}
    }

Writes are atomic (temp file + ``os.replace``) so a crash mid-write can never
corrupt ``state.json`` — important when two staggered cron runs race.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Dict, List

from . import config
from .ml import empty_ml  # single source of truth for a fresh ML node  # noqa: F401

log = logging.getLogger("mind_shot.state")

GLOBAL_KEY = "__global"


def empty_heatmap() -> List[List[List[int]]]:
    """7 weekdays × 24 hours, each cell a [wins, losses] pair."""
    return [[[0, 0] for _ in range(24)] for _ in range(7)]


def empty_global_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "daily_r": {},
        "heatmap": empty_heatmap(),
        "cur_win_streak": 0,
        "cur_loss_streak": 0,
        "max_win_streak": 0,
        "max_loss_streak": 0,
        "journal": [],
        "account_size": config.ACCOUNT_USD,
        "sl_cooldowns": {},
        "last_weekly_summary": None,
    }
    state.update(config.DEFAULT_RISK)
    return state


def load_state() -> Dict[str, Any]:
    """Load state.json, returning {} on first run or unreadable file."""
    if config.STATE_FILE.exists():
        try:
            return json.loads(config.STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as err:
            log.error("state.json unreadable (%s) — starting fresh", err)
    return {}


def save_state(state: Dict[str, Any]) -> None:
    """Atomically persist state to disk."""
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(config.STATE_DIR), prefix=".state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp, config.STATE_FILE)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def ensure_global(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return the global sub-state, healing any missing keys from older files."""
    gs = state.setdefault(GLOBAL_KEY, empty_global_state())
    for key, default in empty_global_state().items():
        gs.setdefault(key, default)
    return gs


def ensure_strategy(state: Dict[str, Any], strategy_id: str) -> Dict[str, Any]:
    """Return the per-strategy sub-state, creating it on first sight."""
    st = state.setdefault(strategy_id, {"active_trade": None, "last_signal_bar": 0, "ml": empty_ml()})
    st.setdefault("active_trade", None)
    st.setdefault("last_signal_bar", 0)
    st.setdefault("ml", empty_ml())
    return st


def load_trained_model() -> Dict[str, Any] | None:
    if config.TRAINED_MODEL_FILE.exists():
        try:
            return json.loads(config.TRAINED_MODEL_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


__all__ = [
    "GLOBAL_KEY",
    "empty_ml",
    "empty_heatmap",
    "empty_global_state",
    "load_state",
    "save_state",
    "ensure_global",
    "ensure_strategy",
    "load_trained_model",
]
