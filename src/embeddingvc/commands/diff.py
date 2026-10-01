"""Compare two immutable committed snapshots."""
from pathlib import Path
from ..diff_engine import compare_snapshots
from ..objects import RepositoryError, find_repository
from ..revisions import manifest, resolve_revision

DiffError = RepositoryError


def diff(old: str, new: str, directory: Path | str = ".") -> str:
    root = find_repository(Path(directory))
    left, right = resolve_revision(root, old), resolve_revision(root, new)
    result = compare_snapshots(root, manifest(root, left), manifest(root, right))
    lines = [f"Comparing {left} -> {right}"]
    for title, key in (("Documents", "documents"), ("Chunks", "chunks"), ("Embedding objects", "embedding_objects")):
        lines.append(f"{title}: " + ", ".join(f"{count} {kind}" for kind, count in result[key].items()))
    lines.append(f"Configuration changed: {result['config_changed']}; effective embedding configuration changed: {result['embedding_config_changed']}")
    storage = result["storage"]
    lines.append(f"Unique referenced object bytes: {storage['old_bytes']} -> {storage['new_bytes']}; additional: {storage['additional_bytes']}")
    lines.append("Ordinal pairing is a structural heuristic, not guaranteed semantic correspondence.")
    if not result["pairs"]:
        lines.append("Vector drift: unavailable (no paired chunks)")
    for pair in result["pairs"]:
        label = f"{pair['document']} [{pair['old_ordinal']} -> {pair['new_ordinal']}, {pair['match']}]"
        if "unavailable" in pair:
            lines.append(f"Vector drift {label}: unavailable ({pair['unavailable']})")
        else:
            lines.append(f"Vector drift {label}: cosine={pair['cosine_similarity']:.6g}, drift={pair['drift']:.6g}, euclidean={pair['euclidean_distance']:.6g}")
    unmatched = result["chunks"]["added"] + result["chunks"]["deleted"]
    if unmatched:
        lines.append(f"Vector drift: unavailable for {unmatched} unpaired chunks")
    return "\n".join(lines)
