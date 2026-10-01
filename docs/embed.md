# Embed

Run `embeddingvc embed` from an initialized collection or a subdirectory.
Install `pip install -e ".[embeddings]"` from the source checkout first, and set
`embedding.revision` to the model's full lowercase 40-character commit SHA.
Branches, tags, short hashes and an unset revision are rejected.

The command locks the repository, rescans all tracked roots, batches only unique
texts missing compatible verified objects, and assigns references to every
occurrence. New files inside tracked directories are included; untracked paths
outside them require `add`. Deleted sources leave the candidate while all
historical objects remain. An empty candidate can represent a complete deletion.

```text
Embedding complete: 6 new objects, 14 reused occurrences
20 chunk occurrences ready
Elapsed: 1.23s
Next: embeddingvc commit -m "Describe this update"
```

This is illustrative output. Reused occurrences include repeated appearances
of a newly generated chunk; generation happens once per unique compatible text.
An unchanged second run calls no encoder batches, although it still loads the
pinned model to validate dimensions and provenance. Use [commit](commit.md) to
publish the prepared snapshot. Embed does not publish new history or synchronize
the database. Like every mutation, it first recovers any interrupted publication.

## Configuration and provenance

`embedding.model`, exact `revision`, `normalize_embeddings` and `batch_size`
are taken from `embeddingvc.yaml`. Optional `pooling` defaults to `model-default`;
supported overrides are `mean`, `max`, `cls`, `mean_sqrt_len_tokens`,
`weightedmean` and `lasttoken`. Overrides require exactly one model Pooling
module. Optional `dimension` defaults to null and verifies the model's reported
dimension when set. Older configs without these optional fields remain valid.

The tokenizer is loaded with the pinned model. Provenance records the model
commit/config hash, tokenizer serialization hash and settings, pooling/module
configuration, sequence length, prompts, library versions, CPU execution and
float32 precision. Model weights use the library cache and never enter history.
The adapter follows the [Sentence Transformers API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).

The effective embedding configuration includes extraction versions,
preprocessing, chunking, model/revision, pooling, normalization, dimension and
provenance. Changing these invalidates vectors, even if chunk text is identical.
Batch size, repository name and database location do not invalidate vectors.

## Storage contract

Version 2 document and occurrence records from `add` remain unchanged. A version
1 empty index is accepted. Hashes use sorted compact UTF-8 JSON with ASCII escapes
and no NaN/Infinity. Existing hashes remain compatible.

The index keeps `config` as its staging snapshot hash. Embed also writes:

- `base_commit`: null if not already present; an existing value is preserved.
- `config_object`: a hash covering the complete validated working configuration.
- `embedding_config_hash` and `embedding_config`: the effective config hash/object.
- `generation`: ready status, object/occurrence counts and elapsed seconds.

Each occurrence's `embedding` reference contains `vector` (the object ID),
`fingerprint` (the lightweight settings check used by add/status), and
`embedding_config_hash` (the complete compatibility identity). Add removes the
generation-ready marker when it changes documents or staging configuration.

Immutable objects live in `.embeddingvc/objects/{chunks,configs,embeddings}/`.
Embedding payloads have this shape:

```json
{
  "schema": 1,
  "chunk_hash": "<chunk object SHA-256>",
  "embedding_config_hash": "<effective configuration SHA-256>",
  "vector": [0.1, 0.2],
  "dimension": 2,
  "provenance": {"provider": "sentence-transformers"}
}
```

Provenance here is abbreviated. No run timestamp is stored in immutable payloads.
The object ID hashes the entire payload. Lookup uses the pair `(chunk_hash,
embedding_config_hash)` and rebuilds from verified objects; no cache is required.
Every embedding object encountered during lookup must be intact. Corrupt or
missing referenced embeddings block publication with an actionable error rather
than silently counting as ready. Existing corrupt objects are never overwritten.

## Failure behavior and tests

Add, config mutation and embed share `.embeddingvc/lock`. Encode failures,
invalid dimensions, wrong result counts, nonnumeric/nonfinite values, corrupt
objects, extraction failures and write failures leave the previous index intact.
Sources, tracked-file membership, YAML bytes, extractor versions and index bytes
are rechecked before the final atomic index replacement. Valid immutable objects
written before failure can be reused on retry.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Unit tests use fake deterministic encoders and mock the library adapter. To opt
into the real-model round trip, install the embedding extra and set an exact SHA:

```powershell
$env:EMBEDDINGVC_REAL_MODEL_REVISION = "<exact-model-commit>"
# Optional: choose a different Sentence Transformers model.
$env:EMBEDDINGVC_REAL_MODEL = "BAAI/bge-small-en-v1.5"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_embed.py -v
```

The integration run may download model weights into the library cache. It also
checks that an unchanged second run never calls encode.
