# Inspect history and compare snapshots

Run these inside an initialized EmbeddingVC collection:

```powershell
embeddingvc log
embeddingvc log --limit 10
embeddingvc diff HEAD~1 HEAD
embeddingvc diff main experiment
```

Revisions accept HEAD, local branch names, full commit IDs, unique prefixes,
and a parent suffix such as HEAD~2. Invalid or ambiguous revisions fail clearly.
`log` follows the current branch ancestry (or detached HEAD), newest first.
An unborn branch prints `No commits yet`. Dates are displayed in UTC, alongside
parent IDs, branch decorations, document/chunk counts and stored statistics.
Existing commits contain snapshot counts but do not retain generation/reuse
statistics; these are shown only when present in a manifest.

`diff` compares committed snapshots only. It verifies manifests and referenced
configuration, chunk and embedding objects. Within each document it pairs exact
chunk content first, preserving duplicate occurrences, then remaining equal
ordinals. Ordinal pairing is a structural heuristic, not a semantic guarantee.
Unmatched occurrences count as additions/deletions. Metadata changes are reported
separately; `metadata_changed` and `stale` are supplementary counts, not additional
occurrences. Unchanged content may be stale when the effective configuration changes.

Each pair reports cosine similarity, drift (1 minus cosine) and Euclidean
distance only when effective embedding configurations match. Equal dimensions
alone do not suffice. Zero-norm vectors, unpaired chunks and incompatible spaces
have unavailable metrics; corrupt/nonfinite objects cause an explicit error.
These measurements do not evaluate retrieval quality.

Storage totals count each unique referenced config/chunk/embedding file once per
snapshot. Additional bytes mean objects referenced only by the new snapshot,
even if already present elsewhere in repository history. Manifest files and
filesystem allocation overhead are excluded.

Both commands are read-only: they do not scan working documents, read working
configuration/index, load models, synchronize Chroma or write reports. CLI errors
use exit 2 for malformed arguments and exit 1 for repository/validation failures.
