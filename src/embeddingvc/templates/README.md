# EmbeddingVC Repository

This folder is managed by EmbeddingVC.

## Getting started

1. Place `.txt`, `.md`, or text-based `.pdf` documents inside `data/`.
2. Review `embeddingvc.yaml`, including the repository name, collection name,
   and immutable embedding model revision (currently unset).

Install the optional embedding dependencies with `pip install -e ".[embeddings]"`
from the EmbeddingVC source checkout. Pin the model revision to a full
40-character commit SHA, then prepare your collection:

```sh
embeddingvc config set model_revision <exact-model-commit>
embeddingvc add data
embeddingvc embed
embeddingvc status
```

`embed` rescans tracked roots and reuses verified compatible vectors. It prepares
the candidate index without committing or updating the database. Commit, log
and checkout are future milestones. DOCX extraction is not yet supported.

## Folder responsibilities

- `data/`: Source documents managed by the user.
- `output/chroma/`: Materialized embedding collection; initially empty.
- `.embeddingvc/`: Internal version history. Do not edit manually.
- `embeddingvc.yaml`: Processing, chunking, and model configuration.

Initialization creates no embeddings, database, or commit. The `main` branch has
no commit yet. Git ignores generated output, cached data, and internal objects.
