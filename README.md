# EmbeddingVC

Git-inspired version control for embedding collections. The first milestone
implements repository initialization using Python 3.11+ and the standard library.

## Local setup (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\embeddingvc.exe init student-notes
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` to use
`embeddingvc` directly. Use `embeddingvc init` for the current directory.
`embeddingvc init <directory> --force` replaces an existing README and config
but always refuses existing `.embeddingvc` metadata. Existing documents and
Git ignore rules are preserved. Linked managed paths are rejected.

Initialization creates `data/`, `output/chroma/`, configuration, instructions,
and `.embeddingvc/` with objects, commits, refs, cache, HEAD and an empty index.
HEAD points to `refs/heads/main`; the empty main reference means no commit yet.
The index starts as `{"version": 1, "documents": {}}`.

The config leaves the model revision unset; pin an immutable model commit before
embedding generation. `status`, `embed`, `commit`, `log`, and `checkout` are future
milestones. No models or database packages are installed by init.

Handled filesystem failures restore overwritten files and remove newly created
files and empty directories. This is not crash recovery; do not run simultaneous
initializers or edit managed files during initialization.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
