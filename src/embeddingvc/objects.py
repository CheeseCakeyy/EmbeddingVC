"""On-disk layout of .embeddingvc: object files and the staging index.

PROVISIONAL CONTRACT. `init` creates an empty index at schema version 1;
`add` is the first command to put anything in it, so the version 2 shape
described below is a proposal for the embed and commit owners to review.

Objects are immutable and content-addressed::

    .embeddingvc/objects/chunks/<content hash>.json    one distinct chunk text
    .embeddingvc/objects/configs/<snapshot hash>.json  one staging snapshot

The index is mutable and names the current staged state::

    {
      "version": 2,
      "tracked_roots": ["data"],              # sorted, none nested in another
      "config": "<snapshot hash>",            # snapshot that produced the below
      "documents": {
        "data/notes.md": {
          "content_hash": "<hash of the raw file bytes>",
          "size": 1234,
          "extractor": "text",
          "occurrences": [
            {"id": ..., "chunk": ..., "ordinal": 0,
             "start": 0, "end": 480, "pages": [1],
             "embedding": null}
          ]
        }
      }
    }

Occurrences are stored in reading order and every appearance is kept, so text
repeated within a document shares one chunk object but keeps one occurrence
per appearance. `embedding` is null until `embed` fills it in.
"""

import json
import os
from pathlib import Path

from .hashing import canonical_json, hash_payload

METADATA_DIRECTORY = ".embeddingvc"
INDEX_FILENAME = "index.json"
INDEX_VERSION = 2
CHUNK_SCHEMA = 1

#: Directories never scanned for documents: internal history, generated output
#: and generated reports. Matched against a path's parts, at any depth.
EXCLUDED_DIRECTORIES = frozenset({METADATA_DIRECTORY, "output", "reports"})

#: Files `init` generates at the repository root. They describe the repository
#: rather than belonging to the collection, so scanning the root must not stage
#: them as documents. Matched at the root only: a README inside data/ is an
#: ordinary document the user put there.
MANAGED_FILES = frozenset({"README.md", "embeddingvc.yaml", ".gitignore"})


class RepositoryError(Exception):
    """The repository is missing, or its metadata cannot be used."""


def find_repository(start: Path) -> Path:
    """Return the repository root at or above `start`.

    Mirrors how Git locates a working tree: walk upwards until the metadata
    directory appears. Stops at the filesystem root.
    """
    current = Path(start).absolute()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / METADATA_DIRECTORY).is_dir():
            return candidate
    raise RepositoryError(
        f"Not an EmbeddingVC repository (no {METADATA_DIRECTORY} directory found in {current} or its parents). "
        "Run 'embeddingvc init' first."
    )


def metadata_directory(root: Path) -> Path:
    return root / METADATA_DIRECTORY


def index_path(root: Path) -> Path:
    return metadata_directory(root) / INDEX_FILENAME


def object_path(root: Path, kind: str, digest: str) -> Path:
    return metadata_directory(root) / "objects" / kind / f"{digest}.json"


def empty_index() -> dict:
    return {"version": INDEX_VERSION, "tracked_roots": [], "config": None, "documents": {}}


def read_index(root: Path) -> dict:
    """Load the staging index, upgrading the empty schema `init` wrote.

    A version 1 index only ever holds an empty `documents` map, so the upgrade
    is a widening: no staged state can be lost by it.
    """
    path = index_path(root)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RepositoryError(f"Index not found: {path}. Run 'embeddingvc init' first.") from exc
    except OSError as exc:
        raise RepositoryError(f"Index cannot be read: {exc}") from exc
    try:
        index = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RepositoryError(f"Index is not valid JSON: {path}: {exc}") from exc
    if not isinstance(index, dict):
        raise RepositoryError(f"Index must be a JSON object: {path}")

    version = index.get("version")
    if version == 1:
        documents = index.get("documents") or {}
        if documents:
            raise RepositoryError(
                f"Index at schema version 1 unexpectedly holds staged documents: {path}. Refusing to upgrade it."
            )
        return empty_index()
    if version != INDEX_VERSION:
        raise RepositoryError(
            f"Index schema version {version!r} is not supported by this version of EmbeddingVC (expected {INDEX_VERSION})."
        )
    for key, default in (("tracked_roots", []), ("documents", {}), ("config", None)):
        index.setdefault(key, default)
    if not isinstance(index["documents"], dict) or not isinstance(index["tracked_roots"], list):
        raise RepositoryError(f"Index is structurally invalid: {path}")
    return index


def _atomic_write(path: Path, payload: bytes) -> None:
    """Replace a file in one step, so a crash leaves the old bytes intact.

    The temporary file is created beside the target because os.replace is only
    atomic within a single filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def write_chunk_object(root: Path, text: str) -> str:
    """Store one chunk's text and return its content hash.

    Writing is skipped when the object already exists: objects are immutable,
    so an existing file with this name already holds exactly this text.
    """
    payload = {"schema": CHUNK_SCHEMA, "text": text, "characters": len(text)}
    digest = hash_payload(payload)
    path = object_path(root, "chunks", digest)
    if not path.exists():
        _atomic_write(path, canonical_json(payload))
    return digest


def write_config_object(root: Path, snapshot: dict) -> str:
    """Store a staging snapshot and return its hash."""
    digest = hash_payload(snapshot)
    path = object_path(root, "configs", digest)
    if not path.exists():
        _atomic_write(path, canonical_json(snapshot))
    return digest


def read_chunk_object(root: Path, digest: str) -> dict:
    path = object_path(root, "chunks", digest)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RepositoryError(f"Chunk object is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RepositoryError(f"Chunk object is corrupt: {path}: {exc}") from exc


def publish_index(root: Path, index: dict) -> None:
    """Replace the index in one step, once every requested file has succeeded.

    Callers build the complete next index in memory and publish here, so a
    failure partway through staging leaves the previous index untouched.
    """
    index = dict(index)
    index["version"] = INDEX_VERSION
    index["tracked_roots"] = sorted(index.get("tracked_roots", []))
    _atomic_write(index_path(root), canonical_json(index) + b"\n")
