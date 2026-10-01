"""Publish a fully embedded candidate without loading an encoder."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .. import commit_manager as manager, config, objects, staging_config
from ..chunking import split_text
from ..document_loader import extractor_versions, load
from ..embedding_engine import effective_configuration
from ..hashing import hash_bytes, hash_payload, occurrence_id
from ..object_store import read_object, reject_links
from ..repository import repository_lock
from .embed import _files, _validate_index


class CommitError(Exception):
    """Publication or synchronization could not complete."""


@dataclass(frozen=True)
class Result:
    commit: str
    branch: str
    documents: int
    occurrences: int
    recovered: bool = False

    def render(self):
        verb = "Synchronized existing" if self.recovered else "Created"
        return (f"{verb} commit {self.commit[:12]} on {self.branch}\n"
                f"{self.documents} documents, {self.occurrences} chunk occurrences\n"
                f"Chroma synchronized: {self.occurrences} records")


def _validate_candidate(root, index):
    _validate_index(index)
    if index.get("generation", {}).get("status") != "ready":
        raise CommitError("Candidate is not ready. Run embeddingvc embed.")
    reject_links(root / "embeddingvc.yaml", root)
    configuration = config.load(root).data
    settings = staging_config.parse(config.dump_yaml(configuration))
    staging = settings.staging_snapshot(extractor_versions(settings.readable_extensions))
    if (read_object(root, "configs", index.get("config")) != staging
            or read_object(root, "configs", index.get("config_object")) != configuration):
        raise CommitError("Configuration changed. Run embeddingvc embed.")
    effective = read_object(root, "configs", index.get("embedding_config"))
    if (effective != effective_configuration(settings, staging, effective["dimension"], effective["provenance"])
            or (settings.embedding.get("dimension") is not None
                and settings.embedding["dimension"] != effective["dimension"])):
        raise CommitError("Effective configuration changed. Run embeddingvc embed.")
    files = _files(root, index["tracked_roots"], index, settings)
    if set(files) != set(index["documents"]):
        raise CommitError("Tracked sources changed. Run embeddingvc embed.")
    for name, path in files.items():
        entry = index["documents"][name]
        raw = path.read_bytes()
        source = load(path, settings, raw=raw)
        chunks = split_text(source.text, settings.chunking.chunk_size, settings.chunking.chunk_overlap)
        if (hash_bytes(raw) != entry["content_hash"] or len(raw) != entry["size"]
                or source.extractor != entry["extractor"] or entry.get("config") != index["config"]
                or len(chunks) != len(entry["occurrences"])):
            raise CommitError("Sources or staged chunks changed. Run embeddingvc embed.")
        for ordinal, (chunk, occurrence) in enumerate(zip(chunks, entry["occurrences"])):
            expected = {"id": occurrence_id(name, ordinal), "ordinal": ordinal,
                        "chunk": hash_payload({"schema": 1, "text": chunk.text, "characters": len(chunk.text)}),
                        "start": chunk.start, "end": chunk.end, "pages": source.pages_for(chunk.start, chunk.end)}
            if (any(occurrence.get(key) != value for key, value in expected.items())
                    or not isinstance(occurrence.get("embedding"), dict)
                    or occurrence["embedding"].get("fingerprint") != settings.embedding_fingerprint()):
                raise CommitError("Staged occurrence is stale. Run embeddingvc embed.")
    return manager.records(root, index)


def _synchronize(root, digest, branch, manifest, adapter_factory, recovered=False):
    try:
        count = manager.sync_commit(root, digest, adapter_factory=adapter_factory)
    except Exception as exc:
        raise CommitError(f"Commit {digest} created; vector-store sync failed. "
                          f"Run embeddingvc checkout HEAD. You can also retry embeddingvc commit -m <message> "
                          f"to synchronize this existing commit. Details: {exc}") from exc
    return Result(digest, branch, len(manifest["documents"]), count, recovered)


def commit(directory: Path | str = ".", *, message: str, adapter_factory=None):
    if not isinstance(message, str) or not message.strip():
        raise CommitError("A nonempty commit message is required.")
    try:
        root = objects.find_repository(Path(directory))
        with repository_lock(root):
            branch, parent = manager.resolve_head(root)
            baseline = manager.read_commit(root, parent) if parent else None
            # Finish a previous synchronization before considering new work.
            # Even if working sources changed, this retry creates no new history.
            sync_path = root / ".embeddingvc/sync.json"
            reject_links(sync_path, root)
            if parent:
                sync = manager.read_json(root, sync_path) if sync_path.exists() else {}
                if sync.get("commit") != parent or sync.get("status") != "synced":
                    return _synchronize(root, parent, branch, baseline, adapter_factory, True)
            reject_links(objects.index_path(root), root)
            index = objects.read_index(root)
            if "base_commit" not in index or index["base_commit"] != parent:
                raise CommitError("Candidate base does not match HEAD. Run embeddingvc embed after restoring the correct index/branch.")
            rows, _ = _validate_candidate(root, index)
            candidate = manager.snapshot(index)
            if baseline and candidate == manager.snapshot(baseline):
                raise CommitError("Nothing to commit.")
            if not baseline and not candidate["documents"]:
                raise CommitError("Nothing to commit.")
            manifest = {"version": 1, "parent": parent, "message": message.strip(),
                        "timestamp": datetime.now(timezone.utc).isoformat(), **candidate,
                        "statistics": {"documents": len(candidate["documents"]), "occurrences": len(rows),
                                       "unique_vectors": len({occ["embedding"]["vector"]
                                           for doc in candidate["documents"].values() for occ in doc["occurrences"]})}}
            # Re-read sources/config/index immediately before making the durable decision.
            _validate_candidate(root, index)
            if objects.read_index(root) != index or manager.resolve_head(root) != (branch, parent):
                raise CommitError("Repository changed during commit. Retry embeddingvc embed.")
            digest = manager.store_commit(root, manifest)
            try:
                manager.publish(root, branch, digest, index)
            except Exception as exc:
                raise CommitError(f"Commit snapshot {digest} stored; publication interrupted. "
                                  f"Retry commit to recover publication before further mutations: {exc}") from exc
            return _synchronize(root, digest, branch, manifest, adapter_factory)
    except CommitError:
        raise
    except Exception as exc:
        raise CommitError(f"{exc}. Run embeddingvc embed if the candidate needs rebuilding.") from exc
