"""Baseline 1: re-embed every chunk occurrence, reusing nothing."""

import time

from .common import chunk_document, encode_all, load_settings, rows_from, source_files


def run(workspace, pool, sync, label: str) -> dict:
    started, cpu = time.perf_counter(), time.process_time()
    settings = load_settings(workspace)
    occurrences = []
    for name, path in source_files(workspace, settings).items():
        occurrences.extend(chunk_document(name, path, settings).occurrences)
    encoder = pool(settings.embedding)
    vectors = encode_all(encoder, [occ.text for occ in occurrences], settings.embedding)
    rows = rows_from(occurrences, vectors)
    sync(workspace, {"persist_directory": "./output/chroma", "collection": f"full-{label}"}).replace(
        f"{label:0<12}", rows, encoder.dimension)
    return {"rows": rows, "seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu,
            "occurrences": len(rows), "reused_occurrences": 0, "changed_documents": None}
