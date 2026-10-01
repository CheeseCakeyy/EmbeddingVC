"""Verified snapshot comparisons with occurrence and physical-storage accounting."""
import math
from collections import Counter, defaultdict, deque

from . import commit_manager
from .object_store import read_object
from .objects import RepositoryError, object_path


def vector_metrics(left, right):
    """Scale before normalizing to avoid overflow in dot products."""
    if len(left) != len(right):
        return {"unavailable": "different dimensions"}
    a, b = max(map(abs, left), default=0), max(map(abs, right), default=0)
    if not a or not b:
        return {"unavailable": "zero-norm vector"}
    x, y = [v / a for v in left], [v / b for v in right]
    nx, ny = math.hypot(*x), math.hypot(*y)
    cosine = max(-1.0, min(1.0, math.fsum((u / nx) * (v / ny) for u, v in zip(x, y))))
    distance = math.dist(left, right)
    if not math.isfinite(distance):
        return {"unavailable": "distance exceeds numeric range"}
    return {"cosine_similarity": cosine, "drift": 1 - cosine, "euclidean_distance": distance}


def _snapshot(root, manifest):
    try:
        rows, _ = commit_manager.records(root, manifest)
        refs = {("configs", manifest[key]) for key in ("config", "config_object", "embedding_config")}
        for doc in manifest["documents"].values():
            for occ in doc["occurrences"]:
                refs.add(("chunks", occ["chunk"]))
                refs.add(("embeddings", occ["embedding"]["vector"]))
        sizes = {}
        for kind, digest in refs:
            read_object(root, kind, digest)
            sizes[kind, digest] = object_path(root, kind, digest).stat().st_size
        return {row["id"]: row["vector"] for row in rows}, sizes
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise RepositoryError(f"Invalid snapshot objects: {exc}") from exc


def compare_snapshots(root, before, after):
    old_vectors, old_sizes = _snapshot(root, before)
    new_vectors, new_sizes = _snapshot(root, after)
    compatible = before["embedding_config_hash"] == after["embedding_config_hash"]
    documents = Counter(dict.fromkeys(("added", "modified", "deleted", "unchanged", "metadata_changed"), 0))
    chunks = Counter(dict.fromkeys(("added", "modified", "deleted", "unchanged", "stale", "metadata_changed"), 0))
    pairs = []
    left_docs, right_docs = before["documents"], after["documents"]
    for name in sorted(left_docs.keys() | right_docs.keys()):
        old, new = left_docs.get(name), right_docs.get(name)
        if old is None or new is None:
            kind = "added" if old is None else "deleted"
            documents[kind] += 1
            chunks[kind] += len((new if old is None else old)["occurrences"])
            continue
        changed = old["content_hash"] != new["content_hash"]
        documents["modified" if changed else "unchanged"] += 1
        metadata_keys = (old.keys() | new.keys()) - {"content_hash", "occurrences", "config"}
        documents["metadata_changed"] += any(old.get(k) != new.get(k) for k in metadata_keys)
        available = defaultdict(deque)
        for index, occ in enumerate(old["occurrences"]):
            available[occ["chunk"]].append(index)
        used, unmatched, matched = set(), [], []
        for occ in new["occurrences"]:
            if available[occ["chunk"]]:
                index = available[occ["chunk"]].popleft()
                used.add(index)
                matched.append((old["occurrences"][index], occ, "content"))
                chunks["unchanged"] += 1
                chunks["stale"] += not compatible
            else:
                unmatched.append(occ)
        remaining = {occ["ordinal"]: occ for i, occ in enumerate(old["occurrences"]) if i not in used}
        for occ in unmatched:
            previous = remaining.pop(occ["ordinal"], None)
            if previous is None:
                chunks["added"] += 1
            else:
                chunks["modified"] += 1
                matched.append((previous, occ, "ordinal heuristic"))
        chunks["deleted"] += len(remaining)
        for previous, occ, method in matched:
            keys = (previous.keys() | occ.keys()) - {"id", "chunk", "embedding"}
            chunks["metadata_changed"] += any(previous.get(k) != occ.get(k) for k in keys)
            metrics = (vector_metrics(old_vectors[previous["id"]], new_vectors[occ["id"]]) if compatible
                       else {"unavailable": "incompatible effective embedding configurations"})
            pairs.append({"document": name, "old_ordinal": previous["ordinal"],
                          "new_ordinal": occ["ordinal"], "match": method, **metrics})
    old_objects = {digest for kind, digest in old_sizes if kind == "embeddings"}
    new_objects = {digest for kind, digest in new_sizes if kind == "embeddings"}
    return {"documents": dict(documents), "chunks": dict(chunks), "pairs": pairs,
            "config_changed": before["config_object"] != after["config_object"],
            "embedding_config_changed": not compatible,
            "embedding_objects": {"shared": len(old_objects & new_objects), "new": len(new_objects - old_objects),
                                  "removed": len(old_objects - new_objects)},
            "storage": {"old_bytes": sum(old_sizes.values()), "new_bytes": sum(new_sizes.values()),
                        "additional_bytes": sum(size for ref, size in new_sizes.items() if ref not in old_sizes)}}
