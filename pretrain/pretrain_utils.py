"""Shared helpers for the pretrain/*.py entrypoints."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


def fgno_root() -> Path:
    return Path(__file__).resolve().parents[1]


def maybe_init_wandb(project: str, run_name: str, enabled: Optional[bool] = None):
    """Initialize wandb unless disabled via FGNO_WANDB=0, or return None if unavailable."""
    if enabled is None:
        enabled = os.environ.get("FGNO_WANDB", "1") != "0"
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        logging.warning("wandb not installed; continuing without logging.")
        return None

    init_kwargs = {
        "project": os.environ.get("FGNO_WANDB_PROJECT", project),
        "name": run_name,
    }
    entity = os.environ.get("FGNO_WANDB_ENTITY") or os.environ.get("WANDB_ENTITY")
    if entity:
        init_kwargs["entity"] = entity
    return wandb.init(**init_kwargs)
