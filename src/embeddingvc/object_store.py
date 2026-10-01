"""Verified, immutable JSON objects shared by staging and embedding generation."""

import json
import math
import os
import re
import tempfile
from pathlib import Path

from .hashing import canonical_json, hash_payload
from .objects import RepositoryError, object_path

KINDS = frozenset({"chunks", "configs", "embeddings"})


def valid_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def reject_links(path: Path, root: Path) -> None:
    """Reject linked metadata, including any parent of the requested path."""
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RepositoryError(f"Path is outside the repository: {path}") from exc
    for candidate in (path, *path.parents):
        if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
            raise RepositoryError(f"Linked path is not supported: {candidate}")
        if candidate == root:
            break


def _path(root: Path, kind: str, digest: str) -> Path:
    if kind not in KINDS or not valid_digest(digest):
        raise RepositoryError("Invalid object kind or SHA-256 identifier")
    path = object_path(root, kind, digest)
    reject_links(path, root)
    return path


def read_object(root: Path, kind: str, digest: str) -> dict:
    path = _path(root, kind, digest)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or hash_payload(payload) != digest:
            raise ValueError("content hash does not match its filename")
        _validate_payload(kind, payload)
        return payload
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise RepositoryError(f"Missing/corrupt {kind} object {digest}: {exc}") from exc


def write_object(root: Path, kind: str, payload: dict) -> str:
    """Install a flushed object atomically without replacing an existing file."""
    if not isinstance(payload, dict):
        raise RepositoryError("Object payload must be a mapping")
    _validate_payload(kind, payload)
    digest = hash_payload(payload)
    path = _path(root, kind, digest)
    if path.exists():
        if read_object(root, kind, digest) != payload:
            raise RepositoryError(f"Immutable object collision: {digest}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".object-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read_object(root, kind, digest) != payload:
                raise RepositoryError(f"Immutable object collision: {digest}")
    finally:
        Path(temporary).unlink(missing_ok=True)
    return digest


def _validate_payload(kind: str, payload: dict) -> None:
    if kind == "chunks" and (payload.get("schema") != 1
            or not isinstance(payload.get("text"), str)
            or type(payload.get("characters")) is not int
            or payload["characters"] != len(payload["text"])):
        raise RepositoryError("Invalid chunk payload")
    if kind == "embeddings":
        validate_embedding(payload)


def validate_vector(values: object, dimension: int) -> list[float]:
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise RepositoryError("Embedding dimension must be a positive integer")
    if not isinstance(values, list) or len(values) != dimension:
        raise RepositoryError(f"Embedding dimension mismatch: expected {dimension} values")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise RepositoryError("Embedding vector contains nonfinite or nonnumeric values")
    return [float(v) for v in values]


def validate_embedding(payload: dict) -> None:
    if (payload.get("schema") != 1 or not valid_digest(payload.get("chunk_hash"))
            or not valid_digest(payload.get("embedding_config_hash"))
            or not isinstance(payload.get("provenance"), dict) or not payload["provenance"]):
        raise RepositoryError("Invalid embedding object payload")
    validate_vector(payload.get("vector"), payload.get("dimension"))


def find_embeddings(root: Path, config_hash: str, dimension: int, provenance: dict) -> dict[str, str]:
    """Rebuild the lookup from verified objects; no cache is authoritative."""
    directory = root / ".embeddingvc" / "objects" / "embeddings"
    reject_links(directory, root)
    found = {}
    if not directory.exists():
        return found
    for path in sorted(directory.glob("*.json")):
        payload = read_object(root, "embeddings", path.stem)
        if payload["embedding_config_hash"] == config_hash:
            if payload["dimension"] != dimension or payload["provenance"] != provenance:
                raise RepositoryError(f"Embedding provenance/dimension mismatch: {path.stem}")
            found.setdefault(payload["chunk_hash"], path.stem)
    return found
