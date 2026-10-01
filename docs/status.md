# Status

Run `embeddingvc status` from an initialized repository or any subdirectory.
It reads sources and metadata without staging, generating embeddings, loading a
model, writing objects, or modifying the database.

Documents and chunks are compared with HEAD; an unborn branch has an empty
baseline. A separate comparison reports changes since the index was staged.
Tracked directories include new supported files discovered inside them.
Supported files under the configured source directory outside tracked scope
are reported as untracked with an `add` hint.

Chunk matching first consumes equal content hashes, including duplicates, within
each document. Remaining old/new occurrences are paired as modified; excess
occurrences are deleted/added. Equal content with an incompatible saved embedding
fingerprint is stale. Counts are occurrences; unique model inputs deduplicate
the pending content hashes. File renames are additions and deletions.

## Storage contracts

Embed uses the contract in [embed.md](embed.md). Commit remains a future command;
its status fixtures use the snapshot shape below:

- `.embeddingvc/commits/<sha256>` is canonical-JSON-addressed snapshot data with
  `documents` and `config`, using the same document records as index version 2.
  The commit filename must equal `hash_payload(snapshot)`.
- An occurrence's embedding reference is `{ "vector": "<sha256>",
  "fingerprint": "<embedding fingerprint>", "embedding_config_hash": "<sha256>" }`.
- `.embeddingvc/objects/embeddings/<sha256>.json` contains `chunk_hash`,
  `embedding_config_hash`, `vector`, `dimension` and reproducibility provenance.
  Status verifies its hash, vector dimension and finite values, full config
  references, and effective model/pipeline configuration without loading a model.
  Older synthetic `objects/vectors/` records remain readable for compatibility.
- Chunk and config objects must exist and match their content-addressed names
  before associated vectors count as ready. A missing or corrupt reference
  requires regeneration; status never repairs it.
- Optional `.embeddingvc/sync.json` contains a `status` of `pending`, `failed`,
  or `synced`. Absence means no synchronization state has been recorded.

Coordinate the commit contract with its implementation. Status fails
clearly for an invalid HEAD snapshot rather than assuming an empty baseline.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```
