"""CLI handlers for embeddingvc configuration."""

from __future__ import annotations

from pathlib import Path

from ..config import ConfigurationError, dump_yaml, get_value, load, set_value


def show(root: Path | str = ".") -> str:
    return dump_yaml(load(root).data).rstrip()


def get(root: Path | str, key: str) -> str:
    value = get_value(root, key)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def set_(root: Path | str, key: str, value: str) -> str:
    before = load(root).effective_hash
    old, new = set_value(root, key, value)
    from ..config import canonical_key
    label = canonical_key(key)
    message = f"Updated {label}: {old} -> {new}"
    if load(root).effective_hash != before:
        message += "\nEmbedding configuration changed. Run embeddingvc status, then embeddingvc embed."
    return message
