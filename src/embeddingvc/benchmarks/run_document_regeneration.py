"""Baseline 2: re-embed every chunk of each changed document.

A document counts as changed when its bytes differ from the baseline or it is
new. A configuration change that alters chunks or vectors affects every
document. Unchanged documents keep their baseline vectors.
"""

import time

from ..hashing import hash_bytes
from .common import chunk_document, encode_all, load_settings, rows_from, source_files


def run(workspace, pool, sync, label: str, baseline: dict, config_changed: bool) -> dict:
    started, cpu = time.perf_counter(), time.process_time()
    settings = load_settings(workspace)
    previous_hashes, previous_rows = baseline["content_hashes"], baseline["rows_by_source"]
    rows, pending, changed = [], [], 0
    for name, path in source_files(workspace, settings).items():
        raw = path.read_bytes()
        if not config_changed and previous_hashes.get(name) == hash_bytes(raw):
            rows.extend(previous_rows.get(name, []))
            continue
        changed += 1
        pending.extend(chunk_document(name, path, settings, raw).occurrences)
    encoder = pool(settings.embedding)
    reused = len(rows)
    rows.extend(rows_from(pending, encode_all(encoder, [occ.text for occ in pending], settings.embedding)))
    rows.sort(key=lambda row: (row["source"], row["metadata"]["ordinal"]))
    sync(workspace, {"persist_directory": "./output/chroma", "collection": f"document-{label}"}).replace(
        f"{label:0<12}", rows, encoder.dimension)
    return {"rows": rows, "seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu,
            "occurrences": len(rows), "reused_occurrences": reused, "changed_documents": changed}
