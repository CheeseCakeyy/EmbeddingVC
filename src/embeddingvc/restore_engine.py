"""Restore verified snapshots with recoverable metadata publication.

A replacement collection is verified before the durable checkout decision.
After that decision, every mutator finishes publication under the shared lock.
Old collections remain available; source files are never restored or deleted.
"""
import re
from pathlib import Path

from . import commit_manager as manager, config, objects, revisions, staging_config
from .embedding_engine import effective_configuration, validate_revision
from .hashing import canonical_json, hash_bytes, occurrence_id
from .object_store import read_object, reject_links

TRANSACTION = ".embeddingvc/transactions/checkout.json"
FILES = ("embeddingvc.yaml", ".embeddingvc/index.json", ".embeddingvc/HEAD", ".embeddingvc/sync.json")


def _relative(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or Path(value).is_absolute() or ".." in Path(value).parts):
        raise objects.RepositoryError("Invalid tracked path in snapshot")


def validate_target(root, digest):
    """Validate stored configuration and graph without reading working sources."""
    value = revisions.manifest(root, digest)
    configuration = read_object(root, "configs", value["config_object"])
    config.validate(configuration, root)
    settings = staging_config.parse(config.dump_yaml(configuration))
    validate_revision(settings.embedding)
    staging = read_object(root, "configs", value["config"])
    effective = read_object(root, "configs", value["embedding_config"])
    # Use the recorded extractor versions, not this machine's installed versions.
    if (staging.get("preprocessing") != configuration["preprocessing"]
            or staging.get("chunking") != configuration["chunking"]
            or effective != effective_configuration(settings, staging, effective["dimension"], effective["provenance"])
            or (settings.embedding.get("dimension") is not None
                and settings.embedding["dimension"] != effective["dimension"])):
        raise objects.RepositoryError("Stored pipeline/configuration mismatch")
    if not isinstance(value["tracked_roots"], list):
        raise objects.RepositoryError("Invalid tracked roots")
    for name in [*value["tracked_roots"], *value["documents"]]:
        _relative(name)
    for name, doc in value["documents"].items():
        if doc.get("config") != value["config"]:
            raise objects.RepositoryError(f"Document configuration mismatch: {name}")
        for ordinal, occ in enumerate(doc["occurrences"]):
            if (occ["ordinal"] != ordinal or occ.get("id") != occurrence_id(name, ordinal)
                    or type(occ.get("start")) is not int or type(occ.get("end")) is not int
                    or not 0 <= occ["start"] <= occ["end"] or not isinstance(occ.get("pages"), list)
                    or any(type(page) is not int or page < 1 for page in occ["pages"])
                    or not isinstance(occ.get("embedding"), dict)
                    or occ["embedding"].get("fingerprint") != settings.embedding_fingerprint()):
                raise objects.RepositoryError(f"Invalid committed occurrence: {name}")
    rows, dimension = manager.records(root, value)
    store = Path(configuration["vector_store"]["persist_directory"])
    store = store if store.is_absolute() else root / store
    reject_links(store, root)
    store = store.resolve()
    if store == root or (root / ".embeddingvc") in (store, *store.parents):
        raise objects.RepositoryError("Vector store cannot overwrite repository metadata")
    protected = [root / configuration["documents"]["source_directory"],
                 *[root / name for name in value["tracked_roots"]],
                 *[root / name for name in value["documents"]]]
    for source in protected:
        source = source.resolve()
        if source != root and (source == store or source in store.parents or store in source.parents):
            raise objects.RepositoryError("Vector store overlaps tracked source paths")
    return value, configuration, settings, rows, dimension


def capture(root):
    result = {}
    for name in FILES:
        path = root / name
        reject_links(path, root)
        result[name] = path.read_bytes().hex() if path.exists() else None
    return result


def require_clean(root, current):
    """Check staging, working YAML and tracked source bytes against current HEAD."""
    from .commands.embed import _files
    if current is None:
        raise objects.RepositoryError("Current branch has no snapshot; use --force to replace staged/configuration state")
    value = revisions.manifest(root, current)
    reject_links(objects.index_path(root), root)
    index = objects.read_index(root)
    reject_links(root / "embeddingvc.yaml", root)
    try:
        configuration = config.load(root).data
        if (index.get("base_commit") != current or manager.snapshot(index) != manager.snapshot(value)
                or configuration != read_object(root, "configs", value["config_object"])):
            raise objects.RepositoryError("Staged or configuration changes")
        settings = staging_config.parse(config.dump_yaml(configuration))
        files = _files(root, value["tracked_roots"], index, settings)
        if ({name: hash_bytes(path.read_bytes()) for name, path in files.items()}
                != {name: doc["content_hash"] for name, doc in value["documents"].items()}):
            raise objects.RepositoryError("Tracked source changes")
    except (KeyError, config.ConfigurationError, staging_config.ConfigError, objects.RepositoryError) as exc:
        raise objects.RepositoryError(f"Dirty checkout: {exc}. Save your work or use --force; source files will be preserved.") from exc


def restored_index(value, digest):
    return {"version": objects.INDEX_VERSION, **manager.snapshot(value), "base_commit": digest,
            "generation": {"status": "ready", "restored_from": digest}}


def _next_files(value, configuration, digest, head, sync):
    return {"embeddingvc.yaml": config.dump_yaml(configuration).encode().hex(),
            ".embeddingvc/index.json": (canonical_json(restored_index(value, digest)) + b"\n").hex(),
            ".embeddingvc/HEAD": (head + "\n").encode().hex(),
            ".embeddingvc/sync.json": (canonical_json(sync) + b"\n").hex()}


def recover_checkout(root):
    """Finish a prepared checkout; conflicts stop all mutators without overwrites."""
    marker = root / TRANSACTION
    reject_links(marker, root)
    if not marker.exists():
        return
    try:
        transaction = manager.read_json(root, marker)
        digest = transaction["commit"]
        value, configuration, _, rows, dimension = validate_target(root, digest)
        head = transaction["head"]
        if head != digest:
            if not re.fullmatch(r"ref: refs/heads/[A-Za-z0-9][A-Za-z0-9._-]*", head):
                raise objects.RepositoryError("Invalid checkout HEAD")
            if revisions.read_text(root, root / ".embeddingvc" / head[5:]) != digest:
                raise objects.RepositoryError("Target branch changed during checkout")
        sync = transaction["sync"]
        if (sync.get("commit") != digest or sync.get("status") != "synced"
                or sync.get("records") != len(rows) or sync.get("dimension") != dimension
                or sync.get("persist_directory") != configuration["vector_store"]["persist_directory"]
                or not isinstance(sync.get("collection"), str) or not sync["collection"]):
            raise objects.RepositoryError("Invalid prepared synchronization record")
        desired = _next_files(value, configuration, digest, head, sync)
        previous = transaction["previous"]
        if transaction.get("version") != 1 or set(previous) != set(FILES):
            raise objects.RepositoryError("Invalid checkout transaction")
        pending = (canonical_json({"commit": digest, "status": "pending"}) + b"\n").hex()
        current = capture(root)
        for name in FILES:
            if previous[name] is not None:
                bytes.fromhex(previous[name])
            allowed = [previous[name], desired[name]]
            if name == ".embeddingvc/sync.json":
                allowed.append(pending)
            if current[name] not in allowed:
                raise objects.RepositoryError(f"Checkout conflicts with edits to {name}")
        # Ready is always published last, after HEAD/index/YAML agree.
        if current != desired:
            manager.write_json(root, root / ".embeddingvc/sync.json", {"commit": digest, "status": "pending"})
            for name in FILES:
                objects._atomic_write(root / name, bytes.fromhex(desired[name]))
        marker.unlink()
    except Exception as exc:
        raise objects.RepositoryError(f"Interrupted checkout needs recovery: {exc}. Retry checkout after resolving the conflict; transaction retained.") from exc


def restore(root, revision, *, force=False, adapter_factory=None):
    digest = revisions.resolve_revision(root, revision)
    value, configuration, _, rows, dimension = validate_target(root, digest)
    if revision == "HEAD":
        count = manager.sync_commit(root, digest, adapter_factory=adapter_factory)
        return digest, revisions.head(root)[0], count, True
    _, current = revisions.head(root)
    if not force:
        require_clean(root, current)
    previous = capture(root)
    ref = root / ".embeddingvc/refs/heads" / revision
    branch = revision if "~" not in revision and ref.is_file() else None
    head = f"ref: refs/heads/{branch}" if branch else digest
    sync = manager.prepare_sync(root, digest, adapter_factory=adapter_factory)
    # Editors do not hold our lock: don't discard changes made while building Chroma.
    if capture(root) != previous:
        raise objects.RepositoryError("Metadata changed during checkout; retry after saving your work")
    if not force:
        require_clean(root, current)
    if branch and revisions.read_text(root, ref) != digest:
        raise objects.RepositoryError("Target branch changed during checkout")
    marker = root / TRANSACTION
    manager.write_json(root, marker, {"version": 1, "commit": digest, "head": head,
                                    "previous": previous, "sync": sync})
    recover_checkout(root)
    return digest, branch, len(rows), False
