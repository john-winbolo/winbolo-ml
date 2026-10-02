# Copyright (c) 2026 John Morrison.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Utility functions for WinBolo ML."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file and return as dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)
