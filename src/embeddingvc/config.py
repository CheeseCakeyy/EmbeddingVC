"""Load, validate, inspect, and atomically update embeddingvc.yaml."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .models.config import Configuration


class ConfigurationError(Exception):
    """The repository configuration is invalid or cannot be updated."""


ALIASES = {
    "model": "embedding.model",
    "model_revision": "embedding.revision",
    "chunk_size": "chunking.chunk_size",
    "chunk_overlap": "chunking.chunk_overlap",
}
SUPPORTED_PROVIDERS = {"chromadb"}
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_scalar(item) for item in inner.split(",")]
    if value.startswith("{"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ConfigurationError(f"Invalid YAML value: {value}") from exc
    if value in {"null", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_yaml(text: str) -> dict[str, Any]:
    """Parse the deliberately small YAML shape emitted by this project."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if "\t" in raw[:indent]:
            raise ConfigurationError(f"Tabs are not supported in YAML (line {number})")
        line = raw.strip()
        if ":" not in line:
            raise ConfigurationError(f"Expected key/value at line {number}")
        key, value = line.split(":", 1)
        key = key.strip()
        if not _KEY.fullmatch(key):
            raise ConfigurationError(f"Invalid key at line {number}: {key}")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if key in parent:
            raise ConfigurationError(f"Duplicate key: {key}")
        if value.strip():
            parent[key] = _scalar(value)
        else:
            parent[key] = {}
            stack.append((indent, parent[key]))
    return root


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        if not value or re.search(r"[:#\[\]{},]|^\s|\s$", value):
            return json.dumps(value)
        return value
    if isinstance(value, list):
        return "[" + ", ".join(_yaml_scalar(item) for item in value) + "]"
    return str(value)


def dump_yaml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    def emit(mapping: dict[str, Any], indent: int = 0) -> None:
        for key, value in mapping.items():
            prefix = " " * indent + key + ":"
            if isinstance(value, dict):
                lines.append(prefix)
                emit(value, indent + 2)
            else:
                lines.append(prefix + " " + _yaml_scalar(value))
    emit(data)
    return "\n".join(lines) + "\n"


def _get(data: dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigurationError(f"Missing configuration key: {dotted}")
        current = current[part]
    return current


def _set(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = data
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def canonical_key(key: str) -> str:
    return ALIASES.get(key, key)


def _inside(root: Path, configured: str, field: str) -> None:
    if not isinstance(configured, str) or not configured:
        raise ConfigurationError(f"{field} must be a non-empty path")
    path = Path(configured)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ConfigurationError(f"{field} must stay inside the repository") from exc


def validate(data: dict[str, Any], root: Path | str) -> Configuration:
    if not isinstance(data, dict):
        raise ConfigurationError("Configuration must be a mapping")
    root = Path(root).absolute()
    allowed = {
        "repository": {"name"},
        "documents": {"source_directory", "supported_types"},
        "preprocessing": {"unicode_normalization", "remove_extra_whitespace", "lowercase"},
        "chunking": {"strategy", "chunk_size", "chunk_overlap"},
        "embedding": {"model", "revision", "normalize_embeddings", "batch_size"},
        "vector_store": {"provider", "collection", "persist_directory"},
    }
    for section, values in data.items():
        if section not in allowed or not isinstance(values, dict):
            raise ConfigurationError(f"Unknown configuration section: {section}")
        unknown = set(values) - allowed[section]
        if unknown:
            raise ConfigurationError(f"Unknown configuration key: {section}.{sorted(unknown)[0]}")
    required = [
        "repository.name", "documents.source_directory", "documents.supported_types",
        "preprocessing.unicode_normalization", "preprocessing.remove_extra_whitespace",
        "preprocessing.lowercase", "chunking.strategy", "chunking.chunk_size",
        "chunking.chunk_overlap", "embedding.model", "embedding.revision",
        "embedding.normalize_embeddings", "embedding.batch_size", "vector_store.provider",
        "vector_store.collection", "vector_store.persist_directory",
    ]
    for key in required:
        _get(data, key)
    _inside(root, _get(data, "documents.source_directory"), "documents.source_directory")
    _inside(root, _get(data, "vector_store.persist_directory"), "vector_store.persist_directory")
    provider = _get(data, "vector_store.provider")
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigurationError(f"Unsupported vector store provider: {provider}")
    size = _get(data, "chunking.chunk_size")
    overlap = _get(data, "chunking.chunk_overlap")
    batch = _get(data, "embedding.batch_size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ConfigurationError("chunking.chunk_size must be a positive integer")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or not 0 <= overlap < size:
        raise ConfigurationError("chunking.chunk_overlap must satisfy 0 <= overlap < chunk_size")
    if not isinstance(batch, int) or isinstance(batch, bool) or batch <= 0:
        raise ConfigurationError("embedding.batch_size must be a positive integer")
    for key in ("preprocessing.remove_extra_whitespace", "preprocessing.lowercase", "embedding.normalize_embeddings"):
        if not isinstance(_get(data, key), bool):
            raise ConfigurationError(f"{key} must be true or false")
    model = _get(data, "embedding.model")
    revision = _get(data, "embedding.revision")
    if not isinstance(model, str) or not model:
        raise ConfigurationError("embedding.model must be a non-empty string")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ConfigurationError("embedding.revision must be an exact revision or null")
    effective = {
        "preprocessing": data["preprocessing"],
        "chunking": data["chunking"],
        "embedding": {key: data["embedding"][key] for key in ("model", "revision", "normalize_embeddings")},
    }
    digest = hashlib.sha256(json.dumps(effective, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return Configuration(data=data, effective_hash=digest)


def load(root: Path | str = ".") -> Configuration:
    root = Path(root).absolute()
    path = root / "embeddingvc.yaml"
    if not path.is_file() or path.is_symlink():
        raise ConfigurationError(f"Missing configuration: {path}")
    return validate(parse_yaml(path.read_text(encoding="utf-8")), root)


def set_value(root: Path | str, key: str, raw_value: str) -> tuple[Any, Any]:
    root = Path(root).absolute()
    current = load(root)
    dotted = canonical_key(key)
    old = _get(current.data, dotted)
    if isinstance(old, bool):
        value = _scalar(raw_value)
        if not isinstance(value, bool):
            raise ConfigurationError(f"{dotted} must be true or false")
    elif isinstance(old, int) and not isinstance(old, bool):
        value = _scalar(raw_value)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigurationError(f"{dotted} must be an integer")
    elif old is None:
        value = None if raw_value.lower() in {"null", "~"} else raw_value
    else:
        value = raw_value
    proposed = json.loads(json.dumps(current.data))
    _set(proposed, dotted, value)
    validate(proposed, root)
    path = root / "embeddingvc.yaml"
    fd, temporary = tempfile.mkstemp(prefix=".embeddingvc-config-", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(dump_yaml(proposed))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        try:
            Path(temporary).unlink()
        except OSError:
            pass
        raise ConfigurationError(f"Could not update configuration: {exc}") from exc
    return old, value


def get_value(root: Path | str, key: str) -> Any:
    return _get(load(root).data, canonical_key(key))
