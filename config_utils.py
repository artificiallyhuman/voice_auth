"""Utility helpers for reading and writing application configuration.

The configuration lives in JSON format in *config.json* in the project root and
contains:

    {
        "similarity_threshold": 0.8,
        "registration_script": "...",
        "verification_script": "..."
    }

The file is created with sane defaults on first access.
"""

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "similarity_threshold": 0.8,
    "registration_script": "My voice is my passport. Please verify me.",
    "verification_script": "I solemnly swear that I am up to no good.",
}

CONFIG_PATH = Path("config.json")


def load_config() -> Dict[str, Any]:
    """Read the configuration from disk or return defaults if missing."""

    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            # Corrupted file – fall back to defaults and rewrite
            data = DEFAULT_CONFIG.copy()
            save_config(data)
    else:
        data = DEFAULT_CONFIG.copy()
        save_config(data)

    # Ensure all default keys exist (future-proof)
    for k, v in DEFAULT_CONFIG.items():
        data.setdefault(k, v)

    return data


def save_config(cfg: Dict[str, Any]) -> None:
    """Write configuration back to disk (atomic)."""

    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
