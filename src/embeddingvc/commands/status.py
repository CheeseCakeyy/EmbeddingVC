"""Read-only source comparison and conservative embedding readiness checks."""

import json
import math
import re
from pathlib import Path

from .add import AddError, _reject_links, _resolve, _walk
from .. import objects, staging_config
from ..change_detector import compare
from ..chunking import split_text
from ..document_loader import DocumentError, extractor_versions, load
from ..hashing import hash_bytes, hash_payload
from ..embedding_engine import effective_configuration
from ..object_store import read_object


class StatusError(Exception):
    """Status cannot determine repository state reliably."""


def _read(path, root):
    _reject_links(path, root)
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _object(root, kind, digest):
    if not _digest(digest):
        return None
    try:
        value = _read(objects.object_path(root, kind, digest), root)
        return value if isinstance(value, dict) and hash_payload(value) == digest else None
    except (OSError, ValueError, AddError):
        return None


def _snapshot(value):
    if not isinstance(value, dict) or not isinstance(value.get("documents"), dict):
        raise StatusError("Invalid document snapshot")
    for name, document in value["documents"].items():
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise StatusError("Invalid document path in snapshot")
        if not isinstance(document, dict) or not _digest(document.get("content_hash")):
            raise StatusError(f"Invalid document record: {name}")
        if not isinstance(document.get("occurrences"), list) or any(
            not isinstance(item, dict) or not _digest(item.get("chunk"))
            for item in document["occurrences"]
        ):
            raise StatusError(f"Invalid chunk occurrences: {name}")
    return value


def _baseline(root):
    metadata = root / ".embeddingvc"
    _reject_links(metadata / "HEAD", root)
    head = (metadata / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: refs/heads/"):
        name = head[len("ref: refs/heads/"):]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise StatusError("Invalid HEAD branch reference")
        ref = metadata / "refs" / "heads" / name
        _reject_links(ref, root)
        commit = ref.read_text(encoding="utf-8").strip()
        label = f"On branch {name}"
        if not commit:
            return label, {"documents": {}}
    else:
        commit = head
        label = f"HEAD detached at {commit[:7]}"
    if not _digest(commit):
        raise StatusError("Invalid HEAD commit identifier")
    payload = _read(metadata / "commits" / commit, root)
    if not isinstance(payload, dict) or hash_payload(payload) != commit:
        raise StatusError("HEAD commit is corrupt")
    return label, _snapshot(payload)


def status(directory: Path | str = ".") -> str:
    """Inspect without staging, loading models, or writing runtime files."""
    try:
        return _status(directory)
    except (OSError, ValueError, objects.RepositoryError, staging_config.ConfigError, AddError, DocumentError) as exc:
        raise StatusError(str(exc)) from exc


def _status(directory):
    root = objects.find_repository(Path(directory))
    _reject_links(root / ".embeddingvc", root)
    _reject_links(objects.index_path(root), root)
    index = _snapshot(objects.read_index(root))
    label, head = _baseline(root)
    settings = staging_config.load(root)
    snapshot = settings.staging_snapshot(extractor_versions(settings.readable_extensions))
    config = hash_payload(snapshot)
    fingerprint = settings.embedding_fingerprint()
    skipped, files = [], {}
    roots = index.get("tracked_roots", []) or list(index["documents"])
    for relative in roots:
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise StatusError("Invalid tracked root")
        path = root / relative
        _reject_links(path, root)
        if not path.exists():
            continue
        request = _resolve(str(path), root, index, settings.readable_extensions,
                           settings.declared_extensions, skipped)
        for file in request.files:
            files[file.relative_to(root).as_posix()] = file
    working = {}
    for name, file in sorted(files.items()):
        raw = file.read_bytes()
        document = load(file, settings, raw=raw)
        occurrences = []
        for chunk in split_text(document.text, settings.chunking.chunk_size, settings.chunking.chunk_overlap):
            digest = hash_payload({"schema": objects.CHUNK_SCHEMA, "text": chunk.text, "characters": len(chunk.text)})
            occurrences.append({"chunk": digest})
        working[name] = {"content_hash": hash_bytes(raw), "occurrences": occurrences}

    untracked_files = []
    source = root / settings.source_directory
    _reject_links(source, root)
    if source.is_dir():
        _walk(source.resolve(), root, settings.readable_extensions, settings.declared_extensions, untracked_files, skipped)
    untracked = sorted(file.relative_to(root).as_posix() for file in untracked_files
                       if file.relative_to(root).as_posix() not in working)

    ready, stale, issues = set(), set(), set()
    for state in (head, index):
        for document in state["documents"].values():
            valid_config = _object(root, "configs", document.get("config", state.get("config"))) is not None
            for occurrence in document["occurrences"]:
                digest, embedding = occurrence["chunk"], occurrence.get("embedding")
                chunk = _object(root, "chunks", digest)
                if not valid_config or chunk is None:
                    issues.add(digest)
                    continue
                if not embedding:
                    continue
                if not isinstance(embedding, dict) or embedding.get("fingerprint") != fingerprint:
                    stale.add(digest)
                    continue
                if embedding.get("embedding_config_hash"):
                    try:
                        vector = read_object(root, "embeddings", embedding.get("vector"))
                        read_object(root, "configs", state.get("config_object"))
                        recorded = read_object(root, "configs", state.get("embedding_config"))
                    except objects.RepositoryError:
                        issues.add(digest)
                        continue
                    current_hash = hash_payload(effective_configuration(
                        settings, snapshot, vector["dimension"], vector["provenance"]))
                    if current_hash != vector["embedding_config_hash"]:
                        stale.add(digest)
                    elif (vector["chunk_hash"] == digest
                            and embedding["embedding_config_hash"] == current_hash
                            and state.get("embedding_config_hash") == current_hash
                            and hash_payload(recorded) == current_hash):
                        ready.add(digest)
                    else:
                        issues.add(digest)
                    continue
                vector = _object(root, "vectors", embedding.get("vector"))
                values = vector.get("values") if vector else None
                if (vector and vector.get("chunk") == digest and vector.get("fingerprint") == fingerprint
                        and isinstance(values, list) and values and all(
                            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)):
                    ready.add(digest)
                else:
                    issues.add(digest)
    stale -= ready
    comparison = compare(head["documents"], working, stale)
    pending = [item["chunk"] for doc in working.values() for item in doc["occurrences"] if item["chunk"] not in ready]
    differences = compare(index["documents"], working)
    changed = any(differences[group][kind] for group in ("documents", "chunks") for kind in ("added", "modified", "deleted"))
    config_changed = bool(index["documents"]) and any(
        doc.get("config", index.get("config")) != config for doc in index["documents"].values())
    lines = [label]
    for group in ("documents", "chunks"):
        counts = comparison[group]
        lines.append(group.capitalize() + ": " + ", ".join(f"{n} {kind}" for kind, n in counts.items()))
    lines.append(f"Embeddings required: {len(pending)} occurrences ({len(set(pending))} unique model inputs)")
    if config_changed:
        lines.append("Staging configuration changed since add.")
    if stale:
        lines.append("Embedding configuration changed; matching text has stale vectors.")
    if issues:
        lines.append(f"Missing/corrupt object references: {len(issues)} unique chunks; regeneration required.")
    if changed or config_changed:
        lines.append("Working state differs from index. Run embeddingvc embed to refresh tracked sources.")
    else:
        lines.append("Working state matches index.")
    if untracked:
        lines.append("Untracked documents (run embeddingvc add <path>): " + ", ".join(untracked))
    sync_path = root / ".embeddingvc" / "sync.json"
    if sync_path.exists():
        sync = _read(sync_path, root)
        if not isinstance(sync, dict) or sync.get("status") not in {"pending", "failed", "synced"}:
            raise StatusError("Invalid sync.json status")
        lines.append(f"Database synchronization: {sync['status']}")
    if not any(comparison[g][k] for g in ("documents", "chunks") for k in ("added", "modified", "deleted", "stale") if k in comparison[g]) and not pending and not untracked and not config_changed:
        lines.append("No document or embedding changes.")
    return "\n".join(lines)
