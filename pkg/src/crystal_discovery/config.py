"""Load the frozen configuration.

Kept separate from the code that uses it so that every scientific constant lives in one
YAML file a reader can inspect without reading Python.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "production.yaml"


@dataclass(frozen=True)
class Config:
    preprocessing: dict
    m3: dict
    soap: dict
    ood: dict
    retrieval: dict

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path else DEFAULT_CONFIG
        if not p.is_file():
            raise FileNotFoundError(f"config not found: {p}")
        d = yaml.safe_load(p.read_text())
        return cls(**{k: d[k] for k in ("preprocessing", "m3", "soap", "ood", "retrieval")})
