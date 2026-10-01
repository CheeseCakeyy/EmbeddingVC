"""Read-only revision resolution and validation of committed history."""
import re
from datetime import datetime, timezone
from pathlib import Path

from .commit_manager import read_commit
from .object_store import reject_links, valid_digest
from .objects import RepositoryError


def read_text(root, path):
    reject_links(path, root)
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise RepositoryError(f"Cannot read {path.name}: {exc}") from exc


def head(root):
    value = read_text(root, root / ".embeddingvc/HEAD")
    if value.startswith("ref: refs/heads/"):
        name = value[len("ref: refs/heads/"):]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise RepositoryError("Invalid HEAD branch")
        value = read_text(root, root / ".embeddingvc/refs/heads" / name)
        if not value:
            return name, None
    else:
        name = None
    if not valid_digest(value):
        raise RepositoryError("Invalid HEAD commit identifier")
    return name, value


def commit_ids(root):
    path = root / ".embeddingvc/commits"
    reject_links(path, root)
    if not path.is_dir():
        raise RepositoryError("Missing commits directory")
    return sorted({p.stem if p.suffix == ".json" else p.name for p in path.iterdir()
                   if valid_digest(p.stem if p.suffix == ".json" else p.name)})


def short_id(digest, ids):
    for length in range(7, 65):
        if not any(other != digest and other.startswith(digest[:length]) for other in ids):
            return digest[:length]
    return digest


def manifest(root, digest):
    try:
        value = read_commit(root, digest)
        if value.get("version") != 1:
            raise ValueError("unsupported manifest version")
        if "parent" not in value or (value["parent"] is not None and not valid_digest(value["parent"])):
            raise ValueError("invalid parent identifier")
        if not isinstance(value["message"], str) or not value["message"].strip():
            raise ValueError("invalid commit message")
        date = datetime.fromisoformat(value["timestamp"])
        if date.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        if not isinstance(value["documents"], dict):
            raise ValueError("invalid documents")
        for doc in value["documents"].values():
            if (not isinstance(doc, dict) or not isinstance(doc.get("occurrences"), list)
                    or not valid_digest(doc.get("content_hash"))):
                raise ValueError("invalid document occurrences")
            ordinals = set()
            for occ in doc["occurrences"]:
                if (not isinstance(occ, dict) or type(occ.get("ordinal")) is not int
                        or occ["ordinal"] < 0 or occ["ordinal"] in ordinals
                        or not valid_digest(occ.get("chunk"))):
                    raise ValueError("invalid or duplicate chunk ordinal/reference")
                ordinals.add(occ["ordinal"])
        for key in ("statistics", "generation"):
            if key in value and not isinstance(value[key], dict):
                raise ValueError(f"invalid {key}")
        return value
    except (OSError, ValueError, TypeError, KeyError, RepositoryError) as exc:
        raise RepositoryError(f"Missing/corrupt commit {digest}: {exc}") from exc


def resolve_revision(root: Path, revision: str) -> str:
    match = re.fullmatch(r"([^~]+)(?:~([0-9]+))?", revision)
    if not match:
        raise RepositoryError(f"Invalid revision: {revision!r}")
    base, count = match.groups()
    if base == "HEAD":
        _, digest = head(root)
    elif re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", base):
        ref = root / ".embeddingvc/refs/heads" / base
        reject_links(ref, root)
        if ref.exists():
            digest = read_text(root, ref) or None
        else:
            matches = [item for item in commit_ids(root) if item.startswith(base.lower())]
            if len(matches) != 1:
                reason = "Ambiguous" if matches else "Unknown"
                raise RepositoryError(f"{reason} revision: {base}")
            digest = matches[0]
    else:
        raise RepositoryError(f"Invalid revision: {revision!r}")
    if digest is None:
        raise RepositoryError(f"No commits yet for {base}")
    seen = set()
    for step in range(int(count or 0) + 1):
        if digest in seen:
            raise RepositoryError("Cycle in commit history")
        seen.add(digest)
        value = manifest(root, digest)
        if step == int(count or 0):
            return digest
        digest = value["parent"]
        if digest is None:
            raise RepositoryError(f"Revision {revision} has no requested ancestor")


def utc_date(value):
    return datetime.fromisoformat(value).astimezone(timezone.utc).isoformat()
