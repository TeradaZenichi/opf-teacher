"""Resolve a self-contained or externally owned case configuration."""
from __future__ import annotations

from pathlib import Path


CASE_SCHEMA_VERSION = 1


def resolve_case_source(source: str | Path) -> tuple[Path, Path]:
    resolved = Path(source).expanduser().resolve()
    if resolved.is_dir():
        root = resolved
        config_path = root / "config.json"
    elif resolved.is_file():
        root = resolved.parent
        config_path = resolved
    else:
        raise FileNotFoundError(f"Case source not found: {resolved}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Case configuration not found: {config_path}")
    return root, config_path


def validate_case_config(config: dict) -> None:
    version = config.get("schema_version", CASE_SCHEMA_VERSION)
    if isinstance(version, bool) or version != CASE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported case schema_version {version!r}; "
            f"expected {CASE_SCHEMA_VERSION}"
        )
