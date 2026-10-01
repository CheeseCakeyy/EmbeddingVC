"""Display committed ancestry without scanning sources or opening a database."""
from pathlib import Path
from ..objects import RepositoryError, find_repository
from .. import revisions
from ..object_store import reject_links, valid_digest

LogError = RepositoryError


def log(directory: Path | str = ".", *, limit: int | None = None) -> str:
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise LogError("Limit must be a positive integer")
    root = find_repository(Path(directory))
    branch, current = revisions.head(root)
    if current is None:
        return "No commits yet"
    ids = revisions.commit_ids(root)
    decorations = {}
    heads = root / ".embeddingvc/refs/heads"
    reject_links(heads, root)
    for ref in sorted(heads.iterdir()):
        if ref.name == ".lock":
            continue
        value = revisions.read_text(root, ref)
        if value and not valid_digest(value):
            raise LogError(f"Invalid branch reference: {ref.name}")
        decorations.setdefault(value, []).append(ref.name)
    lines, seen = [], set()
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
        suffix = f" ({', '.join(labels)})" if labels else ""
        docs = value["documents"]
        parent = value["parent"]
        lines.extend([f"commit {revisions.short_id(current, ids)}{suffix}",
                      f"Parent: {revisions.short_id(parent, ids) if parent else '(root)'}",
                      f"Date: {revisions.utc_date(value['timestamp'])}",
                      *[f"    {line}" for line in value["message"].splitlines()],
                      f"    {len(docs)} documents, {sum(len(d['occurrences']) for d in docs.values())} chunks"])
        for key in ("statistics", "generation"):
            if value.get(key):
                lines.append(f"    {key}: " + ", ".join(f"{k}={v}" for k, v in sorted(value[key].items())))
        lines.append("")
        current = parent
    return "\n".join(lines).rstrip()
