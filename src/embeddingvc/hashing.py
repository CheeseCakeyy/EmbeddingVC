"""Content addressing rules shared by add, embed and commit.

PROVISIONAL CONTRACT. The algorithm and the exact byte sequence fed to it
decide every object name in a repository, so changing anything here
invalidates existing history. These rules were written for `embeddingvc add`
and still need the embed owner's agreement before they are treated as fixed.

Three identifiers are defined:

* content hash - names an immutable object (a document's raw bytes, a chunk's
  text, a configuration snapshot). Equal content always yields one object, so
  repeated text is stored once.
* occurrence ID - names one *appearance* of a chunk inside one document.
  Duplicate text shares a content hash but keeps a distinct occurrence per
  appearance, which is what preserves ordering and provenance.
* document ID - names a tracked file. The repository-relative POSIX path is
  used directly so the index stays readable and diffable.
"""

import hashlib
import json

ALGORITHM = "sha256"
"""Hash function name recorded in objects so a future migration can detect it."""

_ENCODING = "utf-8"
_SEPARATOR = b"\x00"


def hash_bytes(data: bytes) -> str:
    """Return the content hash of raw bytes."""
    return hashlib.new(ALGORITHM, data).hexdigest()


def hash_text(text: str) -> str:
    """Return the content hash of text, encoded UTF-8.

    Text is hashed after normalization, so two documents that normalize to the
    same characters share one chunk object even if their source bytes differ.
    """
    return hash_bytes(text.encode(_ENCODING))


def canonical_json(payload: object) -> bytes:
    """Serialize deterministically: sorted keys, no insignificant whitespace.

    Object files are written through this so that re-serializing an unchanged
    object reproduces the bytes its hash was taken over.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(_ENCODING)


def hash_payload(payload: object) -> str:
    """Return the content hash of a JSON-serializable payload."""
    return hash_bytes(canonical_json(payload))


def document_id(relative_path: str) -> str:
    """Return the identifier for a tracked file.

    The repository-relative POSIX path is the identifier. Renaming a file
    therefore reads as a delete plus an add, which is the same behaviour Git
    presents before rename detection.
    """
    return relative_path


def occurrence_id(document: str, ordinal: int) -> str:
    """Return the identifier for the ordinal-th chunk of a document.

    Derived from the document ID and the zero-based position, so re-running
    `add` over unchanged input reproduces every occurrence ID exactly. The
    NUL separator keeps the two fields unambiguous: no path may contain it.
    """
    if ordinal < 0:
        raise ValueError(f"Occurrence ordinal must not be negative: {ordinal}")
    payload = document.encode(_ENCODING) + _SEPARATOR + str(ordinal).encode("ascii")
    return hash_bytes(payload)
