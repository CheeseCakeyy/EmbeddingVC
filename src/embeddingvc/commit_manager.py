"""Immutable snapshots and recoverable, roll-forward ref/index publication."""

import copy
import json
import os
import re
import tempfile
from pathlib import Path

from . import objects
from .hashing import canonical_json, hash_payload
from .object_store import read_object, reject_links, valid_digest

SNAPSHOT_KEYS = ("tracked_roots", "config", "config_object", "embedding_config",
                 "embedding_config_hash", "documents")


def snapshot(index):
    return {key: copy.deepcopy(index[key]) for key in SNAPSHOT_KEYS}


def read_json(root, path):
    reject_links(path, root)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(root, path, payload):
    reject_links(path, root)
    objects._atomic_write(path, canonical_json(payload) + b"\n")


def resolve_head(root):
    metadata = root / ".embeddingvc"
    reject_links(metadata / "HEAD", root)
    head = (metadata / "HEAD").read_text(encoding="utf-8").strip()
    prefix = "ref: refs/heads/"
    if not head.startswith(prefix):
        raise objects.RepositoryError("Detached HEAD: create a branch and use embeddingvc checkout <branch> before committing.")
    branch = head[len(prefix):]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", branch):
        raise objects.RepositoryError("Invalid HEAD branch")
    path = metadata / "refs/heads" / branch
    reject_links(path, root)
    commit = path.read_text(encoding="utf-8").strip() or None
    if commit is not None and not valid_digest(commit):
        raise objects.RepositoryError("Invalid HEAD commit identifier")
    return branch, commit


def read_commit(root, digest):
    if not valid_digest(digest):
        raise objects.RepositoryError("Invalid commit identifier")
    path = root / ".embeddingvc/commits" / f"{digest}.json"
    reject_links(path, root)
    if not path.exists():  # Read the original status contract too.
        path = path.with_suffix("")
    payload = read_json(root, path)
    if not isinstance(payload, dict) or hash_payload(payload) != digest:
        raise objects.RepositoryError("Commit is corrupt")
    return payload


def store_commit(root, manifest):
    digest = hash_payload(manifest)
    path = root / ".embeddingvc/commits" / f"{digest}.json"
    reject_links(path, root)
    if path.exists():
        read_commit(root, digest)
        return digest
    fd, temporary = tempfile.mkstemp(prefix=".commit-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            read_commit(root, digest)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return digest


def recover_publication(root):
    """Called under the shared lock before EVERY mutation; idempotent on retry.

    The durable marker is the publication decision. Until it is removed, all
    mutations must finish its ref, index and pending-sync writes first.
    """
    path = root / ".embeddingvc/transactions/commit.json"
    reject_links(path, root)
    if not path.exists():
        return
    try:
        transaction = read_json(root, path)
        digest = transaction["commit"]
        manifest = read_commit(root, digest)
        branch, current = resolve_head(root)
        if branch != transaction["branch"] or current not in (manifest["parent"], digest):
            raise objects.RepositoryError("HEAD conflicts with interrupted publication")
        index = transaction["index"]
        reject_links(objects.index_path(root), root)
        current_index = objects.read_index(root)
        if (index["base_commit"] != digest or snapshot(index) != snapshot(manifest)
                or hash_payload(current_index) not in (transaction["old_index_hash"], hash_payload(index))):
            raise objects.RepositoryError("Index conflicts with interrupted publication")
        ref = root / ".embeddingvc/refs/heads" / branch
        reject_links(ref, root)
        objects._atomic_write(ref, (digest + "\n").encode())
        reject_links(objects.index_path(root), root)
        objects.publish_index(root, index)
        write_json(root, root / ".embeddingvc/sync.json", {"commit": digest, "status": "pending"})
        path.unlink()
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise objects.RepositoryError(f"Interrupted commit publication needs recovery: {exc}") from exc


def publish(root, branch, digest, index):
    next_index = copy.deepcopy(index)
    next_index["base_commit"] = digest
    write_json(root, root / ".embeddingvc/transactions/commit.json", {
        "version": 1, "branch": branch, "commit": digest,
        "old_index_hash": hash_payload(index), "index": next_index,
    })
    recover_publication(root)


def records(root, manifest):
    """Reconstruct occurrence records entirely from immutable objects."""
    effective = read_object(root, "configs", manifest["embedding_config"])
    if (type(effective.get("dimension")) is not int or effective["dimension"] <= 0
            or not isinstance(effective.get("provenance"), dict) or not effective["provenance"]):
        raise objects.RepositoryError("Invalid effective dimension/provenance")
    if manifest["embedding_config_hash"] != manifest["embedding_config"]:
        raise objects.RepositoryError("Effective configuration hash mismatch")
    result, ids = [], set()
    for name, document in manifest["documents"].items():
        for occurrence in document["occurrences"]:
            identifier = occurrence["id"]
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise objects.RepositoryError("Invalid or duplicate occurrence ID")
            ids.add(identifier)
            chunk = read_object(root, "chunks", occurrence["chunk"])
            reference = occurrence["embedding"]
            vector = read_object(root, "embeddings", reference["vector"])
            if (vector["chunk_hash"] != occurrence["chunk"]
                    or vector["embedding_config_hash"] != manifest["embedding_config_hash"]
                    or reference["embedding_config_hash"] != manifest["embedding_config_hash"]
                    or vector["dimension"] != effective["dimension"]
                    or vector["provenance"] != effective["provenance"]):
                raise objects.RepositoryError("Embedding reference/configuration mismatch")
            result.append({"id": identifier, "vector": vector["vector"], "text": chunk["text"],
                           "metadata": {"source": name, "ordinal": occurrence["ordinal"],
                                        "start": occurrence["start"], "end": occurrence["end"],
                                        "pages": json.dumps(occurrence["pages"]),
                                        "chunk": occurrence["chunk"]}})
    return result, effective["dimension"]


def sync_commit(root, digest, *, adapter_factory=None):
    """Synchronize an existing commit, under repository_lock (checkout contract).

    This never reads working documents, generates vectors or creates history.
    sync.json is the authoritative pointer to the validated physical collection.
    """
    marker = root / ".embeddingvc/sync.json"
    write_json(root, marker, {"commit": digest, "status": "pending"})
    try:
        sync = prepare_sync(root, digest, adapter_factory=adapter_factory)
        write_json(root, marker, sync)
        return sync["records"]
    except Exception as exc:
        try:
            write_json(root, marker, {"commit": digest, "status": "failed", "error": str(exc)})
        except OSError:
            pass  # A durable pending marker is also explicitly not ready.
        raise


def prepare_sync(root, digest, *, adapter_factory=None):
    """Build a verified replacement without changing the active sync pointer."""
    from .vector_store.chromadb_adapter import ChromaAdapter

    manifest = read_commit(root, digest)
    configuration = read_object(root, "configs", manifest["config_object"])
    rows, dimension = records(root, manifest)
    adapter = (adapter_factory or ChromaAdapter)(root, configuration["vector_store"])
    collection = adapter.replace(digest, rows, dimension)
    return {"commit": digest, "status": "synced", "collection": collection,
            "persist_directory": configuration["vector_store"]["persist_directory"],
            "records": len(rows), "dimension": dimension}
