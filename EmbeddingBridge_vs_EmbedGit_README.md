# EmbeddingBridge vs. EmbedGit: Granularity, Computation, and Storage

This document explains the following comparison:

> EmbeddingBridge uses comparatively coarse-grained versioning, which can cause unnecessary regeneration and storage when small source changes produce complete replacement embedding objects. EmbedGit uses chunk-level content addressing and configuration-aware invalidation so each commit represents a complete logical state while physically storing only new or changed objects.

## 1. Coarse-grained versus fine-grained versioning

Granularity means the size of the item treated as one versioned unit.

Suppose a document contains 100 paragraphs.

### Coarse-grained tracking

In a coarse-grained design, the complete document or its resulting vector is treated as one tracked unit:

```text
document.txt -> one tracked embedding object
```

If only paragraph 50 changes, the tracked unit has still changed. The system may therefore create a complete replacement embedding object.

It understands:

> The document's embedding changed.

However, it may not understand:

> Only this small part of the document changed.

### Chunk-level tracking

EmbedGit divides the document into deterministic chunks:

```text
document.txt
|-- chunk 1
|-- chunk 2
|-- ...
|-- chunk 50
`-- chunk 100
```

Each chunk has its own content hash and embedding object. If only chunk 50 changes:

```text
Chunks 1-49:   reuse
Chunk 50:      regenerate
Chunks 51-100: reuse
```

This allows EmbedGit to identify and process the smallest affected unit.

## 2. Why coarse tracking may waste computation

Embedding generation is usually the expensive operation because it requires running an AI model or calling an external API.

Assume that:

- A document contains 100 chunks.
- Only 2 chunks change.
- Embedding one chunk takes 20 milliseconds.

A complete regeneration would require approximately:

```text
100 x 20 ms = 2,000 ms
```

Chunk-level regeneration would require approximately:

```text
2 x 20 ms = 40 ms
```

The theoretical regeneration reduction would be:

```text
1 - (2 / 100) = 98%
```

Actual execution time will also include hashing, document loading, model startup, database synchronization, and other overhead. Therefore, exact efficiency claims must be supported by benchmarks. Nevertheless, this design avoids 98 unnecessary embedding-model calls in this example.

## 3. Why coarse tracking may waste storage

Suppose each embedding vector requires 1 KB.

If all 100 vectors are stored again, the new version requires:

```text
100 vectors x 1 KB = 100 KB
```

If only two vectors changed, chunk-level storage requires only:

```text
2 vectors x 1 KB = 2 KB
```

The remaining 98 vectors are not copied. The new commit refers to embedding objects that already exist.

This is content-addressable storage:

```text
object ID = hash(object content)
```

Identical content produces the same identifier, allowing one immutable object to be shared by multiple commits.

## 4. A complete logical state is not a complete physical copy

This distinction is central to EmbedGit's design.

Every commit should describe the complete collection:

```text
Commit B
|-- chunk 1   -> existing object A1
|-- chunk 2   -> existing object A2
|-- ...
|-- chunk 50  -> new object B50
`-- chunk 100 -> existing object A100
```

Commit B is logically complete because its manifest identifies every object belonging to that version.

Physically, only object `B50` must be added. The unchanged objects already exist and are referenced by the new manifest.

```text
Logical snapshot = complete manifest of references
Physical update  = only new or changed objects
```

This provides straightforward restoration without duplicating the complete embedding collection.

## 5. Why not store only a chain of differences?

A pure delta design could store only instructions such as:

```text
Version B = Version A, except chunks 12 and 50 changed
```

Although this can reduce manifest size, a long delta chain introduces disadvantages:

- Restoring a recent version may require replaying many previous deltas.
- A corrupted or missing delta can affect every later version.
- Garbage collection becomes more complicated.
- Comparing distant versions becomes slower.
- Branching and merging become harder to reason about.
- Floating-point vector deltas may not compress as effectively as expected.

EmbedGit therefore combines the advantages of snapshots and incremental storage:

> Every commit is a complete logical snapshot, but physical storage contains only unique immutable objects.

Object-level delta compression or pack files can be introduced later as an additional optimization without changing these commit semantics.

## 6. Configuration-aware invalidation

Unchanged text does not necessarily mean that its existing embedding remains valid.

Consider this chunk:

```text
Operating systems manage computer resources.
```

Suppose its original configuration is:

```yaml
model: BAAI/bge-small-en-v1.5
model_revision: revision-a
normalize: true
```

The configuration is later changed to:

```yaml
model: sentence-transformers/all-MiniLM-L6-v2
model_revision: revision-b
normalize: true
```

The text is unchanged, but the models produce vectors in different semantic spaces. Reusing the old vector would therefore be unsafe.

EmbedGit checks both the chunk content and its effective embedding configuration:

```text
reusable =
    same chunk hash
    AND same embedding-configuration hash
    AND existing object passes integrity verification
```

If the text is unchanged but the effective configuration changed, the embedding is classified as `STALE` and regenerated.

The effective configuration may include:

- Model name and exact revision
- Tokenizer version
- Pooling method
- Vector dimensions
- Normalization setting
- Text extraction and preprocessing rules
- Chunk size and overlap

This is more reliable than checking only whether the source file changed.

## 7. Chunk change classifications

EmbedGit can classify each chunk and choose the required action:

| Situation | Classification | Action |
|---|---|---|
| A new chunk appears | `ADDED` | Generate a new embedding |
| Existing chunk text changes | `MODIFIED` | Generate a replacement embedding |
| A chunk disappears | `DELETED` | Remove its reference from the new manifest |
| Content and configuration are identical | `UNCHANGED` | Reuse the existing object |
| Content is identical but configuration changed | `STALE` | Regenerate the embedding |

This classification system is the foundation of incremental regeneration.

## 8. Precise comparison with EmbeddingBridge

It would be inaccurate to claim that EmbeddingBridge simply stores everything without optimization. It already contains concepts related to hashing, compression, garbage collection, and content-addressable storage.

The more defensible distinction concerns the size of the tracked unit:

> EmbeddingBridge primarily associates versioned embedding objects with complete source files or vectors. Consequently, a small modification to a large source can invalidate and replace a comparatively large embedding unit, even when most of the source remains unchanged.

EmbedGit instead decomposes documents into deterministic and independently addressable chunks. Each chunk is evaluated using both a content hash and an effective embedding-configuration hash.

Unchanged and compatible chunks retain references to existing immutable embedding objects. Only added, modified, or stale chunks require regeneration.

## 9. Academic formulation

The following paragraph is suitable for a report, synopsis, or presentation:

> EmbeddingBridge primarily associates versioned embedding objects with complete source files or vectors. Consequently, a small modification to a large source may invalidate and replace a comparatively large embedding unit, even when most of the source remains unchanged. EmbedGit instead decomposes documents into deterministic, independently addressable chunks. Each chunk is evaluated using both a content hash and an effective embedding-configuration hash. Unchanged and compatible chunks retain references to existing immutable embedding objects, while only added, modified, or stale chunks require regeneration. A commit still represents the complete logical collection through its manifest, but it does not physically duplicate every vector. This design is expected to reduce embedding-model invocations and version-storage growth for localized changes, subject to experimental validation.

## 10. Short presentation explanation

> EmbeddingBridge mainly tracks changes at a larger unit. EmbedGit tracks individual chunks, stores only new objects, and safely reuses existing objects after checking both content and configuration.

An even shorter summary is:

> **Complete logical snapshots, incremental physical storage.**

## 11. Important research wording

Until controlled experiments have been completed, describe the benefits as expected outcomes:

- "EmbedGit is designed to reduce unnecessary regeneration."
- "The proposed method is expected to reduce storage growth for localized changes."
- "Efficiency improvements will be measured against full and document-level regeneration baselines."

After completing the benchmarks, these statements can be replaced with measured results such as regeneration reduction, elapsed-time improvement, storage overhead, embedding reuse rate, and retrieval-quality changes.
