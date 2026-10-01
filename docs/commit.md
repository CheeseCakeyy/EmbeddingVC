# Commit snapshots

```sh
pip install -e ".[vector-store]"
embeddingvc embed
embeddingvc commit -m "Initial knowledge-base embeddings"
```

Commit validates the complete embedded candidate against the current tracked
files, extraction/chunking results, configuration, and every referenced chunk and
vector. It never loads an embedding model or generates vectors. Run `embed` after
editing sources, changing configuration, or staging different documents.

A nonempty message and an attached branch are required. The candidate's
`base_commit` must match HEAD. Unchanged snapshots and empty initial collections
are rejected as `Nothing to commit.` Deletion-only snapshots, including deletion
of every previously committed document, are supported after `embed`.

## Stored history

`.embeddingvc/commits/<sha256>.json` is an immutable, canonical JSON manifest.
The SHA-256 covers the complete payload: version, parent, message, UTC timestamp,
tracked roots, document/occurrence records, config object identifiers, effective
embedding configuration identifier, and statistics. Vectors and text are
referenced in the shared object store, not copied into each commit. The snapshot
reconstructs the embedding collection; it does not archive original PDF/file bytes.

Commit advances `.embeddingvc/refs/heads/<branch>` and `index.base_commit`.
Before those writes, it persists `.embeddingvc/transactions/commit.json`, including
the intended branch, commit, new index and old index hash. This marker is a durable
decision to finish publication. Every subsequent mutation recovers it under the
shared repository lock, then removes it only after ref, index and pending sync
state are consistent. Conflicting state blocks recovery rather than overwriting
external changes. Do not remove transaction markers as cache cleanup.

The OS-backed `.embeddingvc/mutation.lock` guard releases automatically on process
exit. Its file stays on disk and must not be removed. The temporary `lock` sentinel
is removed normally; a recognized sentinel left by a dead process is reclaimed
while holding the guard. Unknown legacy lock files are conservatively retained.
Individual files are flushed before atomic replacement; this is process-crash
recovery, not a guarantee against every filesystem or power-loss failure.

## Chroma synchronization and retry

Chroma is imported lazily. Every insertion supplies explicit stored vectors,
occurrence IDs, chunk text, and source/ordinal/offset/page metadata. The adapter
builds a uniquely named physical collection under the configured persist directory
and verifies exact membership, dimensions, vector values, text, and metadata.
The configured collection name supplies its namespace. Only after verification
does `.embeddingvc/sync.json` become `synced`, with the commit ID, physical
collection name, persist directory, record count and dimension. Consumers must
use that pointer, not guess a physical collection name from the YAML config.
Empty replacement collections are valid. Old collections remain intact; automatic
garbage collection of old or failed replacements is outside this command's scope.

If synchronization fails, history remains committed, the process returns nonzero,
and sync state remains `pending` or becomes `failed`. The error includes the full
commit ID and the planned `embeddingvc checkout HEAD` recovery instruction.
**Checkout is not implemented yet.** For recovery now, rerun
`embeddingvc commit -m "Retry synchronization"`: it synchronizes the existing HEAD,
without creating another commit, even if working files have since changed. The
retry message is not stored. A later invocation can publish newly embedded work.

The shared checkout integration API is
`commit_manager.sync_commit(root, commit_id, adapter_factory=None)`. Call it under
`repository_lock(root)`. It reads only immutable history/objects and returns the
record count; it neither restores working files nor advances refs.

## Validation

Run `python -m unittest discover -s tests -v`. `test_commit.py` covers complete
snapshots, reuse, no-ops, deletions, stale/corrupt references, CLI errors, publication
failures, synchronization failures, recovery conflicts and actual process death.
`test_vector_store.py` exercises explicit vectors, batched verification and empty
replacement collections. Its real Chroma subprocess test runs when the optional
vector-store dependency is installed. No model download is needed for these tests.
