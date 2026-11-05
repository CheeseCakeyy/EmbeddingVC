"""Display committed ancestry without scanning sources or opening a database."""
from pathlib import Path
from ..objects import RepositoryError, find_repository
from .. import revisions
from ..object_store import reject_links, valid_digest

LogError = RepositoryError


def _branches(root: Path) -> dict[str, str]:
    heads = root / ".embeddingvc/refs/heads"
    reject_links(heads, root)
    tips = {}
    for ref in sorted(heads.iterdir()):
        if ref.name == ".lock":
            continue
        value = revisions.read_text(root, ref)
        if value and not valid_digest(value):
            raise LogError(f"Invalid branch reference: {ref.name}")
        tips[ref.name] = value
    return tips


def graph(directory: Path | str = ".") -> dict:
    """Return every commit reachable from a branch or HEAD, for drawing the tree."""
    root = find_repository(Path(directory))
    branch, current = revisions.head(root)
    tips = {name: value for name, value in _branches(root).items() if value}
    ids = revisions.commit_ids(root)
    commits = {}
    for digest in [*tips.values(), current]:
        while digest and digest not in commits:
            value = revisions.manifest(root, digest)
            commits[digest] = {"commit": digest, "short": revisions.short_id(digest, ids), "parent": value["parent"],
                               "message": value["message"].splitlines()[0], "timestamp": value["timestamp"]}
            digest = value["parent"]
    return {"branch": branch, "head": current, "tips": tips, "commits": list(commits.values())}


def history(directory: Path | str = ".", *, limit: int | None = None) -> list[dict]:
    """Return committed ancestry as structured entries, newest first."""
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise LogError("Limit must be a positive integer")
    root = find_repository(Path(directory))
    branch, current = revisions.head(root)
    if current is None:
        return []
    ids = revisions.commit_ids(root)
    decorations = {}
    for name, value in _branches(root).items():
        decorations.setdefault(value, []).append(name)
    entries, seen = [], set()
    while current is not None and (limit is None or len(seen) < limit):
        if current in seen:
            raise LogError(f"Cycle in commit history at {current}")
        seen.add(current)
        value = revisions.manifest(root, current)
        labels = decorations.get(current, []).copy()
        if len(seen) == 1:
            if branch in labels:
                labels.remove(branch)
            labels.insert(0, f"HEAD -> {branch}" if branch else "HEAD")
        docs = value["documents"]
        parent = value["parent"]
        occurrences = [occurrence for doc in docs.values() for occurrence in doc["occurrences"]]
        entries.append({"commit": current, "short": revisions.short_id(current, ids), "labels": labels,
                        "parent": parent, "parent_short": revisions.short_id(parent, ids) if parent else None,
                        "date": revisions.utc_date(value["timestamp"]), "message": value["message"],
                        "documents": len(docs), "chunks": len(occurrences),
                        "vectors": {occurrence["embedding"].get("vector") for occurrence in occurrences
                                    if isinstance(occurrence.get("embedding"), dict)},
                        "statistics": value.get("statistics"), "generation": value.get("generation")})
        current = parent
    return entries


def log(directory: Path | str = ".", *, limit: int | None = None) -> str:
    entries = history(directory, limit=limit)
    if not entries:
        return "No commits yet"
    lines = []
    for entry in entries:
        suffix = f" ({', '.join(entry['labels'])})" if entry["labels"] else ""
        lines.extend([f"commit {entry['short']}{suffix}",
                      f"Parent: {entry['parent_short'] or '(root)'}",
                      f"Date: {entry['date']}",
                      *[f"    {line}" for line in entry["message"].splitlines()],
                      f"    {entry['documents']} documents, {entry['chunks']} chunks"])
        for key in ("statistics", "generation"):
            if entry[key]:
                lines.append(f"    {key}: " + ", ".join(f"{k}={v}" for k, v in sorted(entry[key].items())))
        lines.append("")
    return "\n".join(lines).rstrip()
