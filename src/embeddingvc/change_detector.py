"""Pure, deterministic content-first comparison of document snapshots."""

from collections import Counter


def compare(before: dict, after: dict, stale: set[str] | None = None) -> dict:
    """Match identical content before pairing remaining chunks within each file.

    Duplicate occurrences are counted individually. A boundary shift cannot
    turn an identical later chunk into a modification just by moving its ordinal.
    ``stale`` contains chunk hashes whose existing vectors are incompatible.
    """
    documents = dict.fromkeys(("added", "modified", "deleted", "unchanged"), 0)
    chunks = dict.fromkeys(("added", "modified", "deleted", "unchanged", "stale"), 0)
    for name in sorted(before.keys() | after.keys()):
        old, new = before.get(name), after.get(name)
        kind = ("added" if old is None else "deleted" if new is None else
                "unchanged" if old["content_hash"] == new["content_hash"] else "modified")
        documents[kind] += 1
        left = Counter(item["chunk"] for item in (old or {}).get("occurrences", []))
        right = Counter(item["chunk"] for item in (new or {}).get("occurrences", []))
        matched = left & right
        for digest, count in matched.items():
            chunks["stale" if digest in (stale or set()) else "unchanged"] += count
        removed, added = sum((left - matched).values()), sum((right - matched).values())
        modified = min(removed, added)
        chunks["modified"] += modified
        chunks["deleted"] += removed - modified
        chunks["added"] += added - modified
    return {"documents": documents, "chunks": chunks}
