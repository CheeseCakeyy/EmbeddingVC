"""Proposed method: the real `embeddingvc embed` and `commit` commands.

Timing therefore includes everything EmbeddingVC does on an update -
rescanning, hashing, verifying stored objects and publishing a commit - not
only the model calls it avoids.
"""

import time

from .. import commit_manager as manager, objects
from ..commands.commit import commit
from ..commands.embed import embed


def run(workspace, pool, sync, label: str) -> dict:
    started, cpu = time.perf_counter(), time.process_time()
    result = embed(workspace, encoder_factory=pool)
    embedded = time.perf_counter()
    _, head = manager.resolve_head(workspace)
    committed = False
    # A no-op update (e.g. metadata only) has nothing to commit; HEAD stays valid.
    if manager.snapshot(objects.read_index(workspace)) != manager.snapshot(manager.read_commit(workspace, head)):
        head = commit(workspace, message=f"benchmark update {label}", adapter_factory=sync).commit
        committed = True
    finished = time.perf_counter()
    records, _ = manager.records(workspace, manager.read_commit(workspace, head))
    rows = [{"id": row["id"], "source": row["metadata"]["source"], "text": row["text"],
             "vector": row["vector"], "metadata": row["metadata"]} for row in records]
    return {"rows": rows, "seconds": finished - started, "cpu_seconds": time.process_time() - cpu,
            "embed_seconds": embedded - started, "commit_seconds": finished - embedded,
            "occurrences": result.occurrences, "reused_occurrences": result.reused,
            "changed_documents": None, "commit": head, "committed": committed}
