# EmbeddingVC Repository

This folder is managed by EmbeddingVC.

## Getting started

1. Place `.txt`, `.md`, `.pdf`, or `.docx` documents inside `data/`.
2. Review `embeddingvc.yaml`, including the repository name, collection name,
   and immutable embedding model revision (currently unset).

Only `embeddingvc init` is implemented in this milestone. The planned workflow is:

```sh
embeddingvc status
embeddingvc embed
embeddingvc commit -m "Initial embedding collection"
embeddingvc log
embeddingvc checkout <commit-id>
```

## Folder responsibilities

- `data/`: Source documents managed by the user.
- `output/chroma/`: Materialized embedding collection; initially empty.
- `.embeddingvc/`: Internal version history. Do not edit manually.
- `embeddingvc.yaml`: Processing, chunking, and model configuration.

Initialization creates no embeddings, database, or commit. The `main` branch has
no commit yet. Git ignores generated output, cached data, and internal objects.
