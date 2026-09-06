# EmbeddingVC

Git-inspired version control for embedding collections. The first milestone
implements repository initialization using Python 3.11+ and the standard library.

## Requirements

- Python 3.11 or newer
- Git

EmbeddingVC currently uses only the Python standard library. Installing it in a
virtual environment keeps each contributor's machine isolated and reproducible.

## Install for development

Clone the repository, enter it, and create a virtual environment.

```powershell
git clone <github-repository-url>
cd embeddingVC
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

The `-e` editable installation means local source-code changes take effect
without reinstalling the package. Verify the installation:

```powershell
.\.venv\Scripts\embeddingvc.exe --version
.\.venv\Scripts\embeddingvc.exe --help
```

On macOS or Linux, use:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/embeddingvc --help
```

## Try the init command safely

Create a disposable project outside the source repository. From the
`embeddingVC` directory on PowerShell:

```powershell
$testDirectory = Join-Path $env:TEMP "embeddingvc-manual-test"
New-Item -ItemType Directory -Path $testDirectory -Force | Out-Null
Set-Location $testDirectory
& "<path-to-embeddingVC>\.venv\Scripts\embeddingvc.exe" init student-notes
Get-ChildItem -Force .\student-notes
Get-Content .\student-notes\.embeddingvc\HEAD
Get-Content .\student-notes\.embeddingvc\index.json
```

Replace `<path-to-embeddingVC>` with the absolute path where you cloned this
repository. A successful run creates `student-notes` and prints the next steps.
Its `HEAD` should contain `ref: refs/heads/main`, and `index.json` should contain
an empty `documents` object.

Run the same command a second time to test overwrite protection:

```powershell
& "<path-to-embeddingVC>\.venv\Scripts\embeddingvc.exe" init student-notes
```

The second run should exit with an error stating that the repository is already
initialized. Do not use an important existing directory for manual tests.

Use `embeddingvc init` for the current directory.
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

Run the automated test suite before pushing changes:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

macOS and Linux:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The suite tests initialization, existing-file protection, `--force`, Git ignore
preservation, invalid targets, rollback after a simulated failure, and CLI error
handling. A successful run currently reports `Ran 9 tests` followed by `OK`.

## Team workflow

The repository owner should create an empty GitHub repository without adding a
README, license, or `.gitignore`, because those files already exist locally. Then
connect and push this checkout:

```powershell
git branch -M main
git remote add origin <github-repository-url>
git push -u origin main
```

Add the other three contributors under the GitHub repository's collaborator or
team settings. Each contributor follows the installation steps above.

For each task, create a short-lived branch from the latest `main`:

```powershell
git switch main
git pull --ff-only
git switch -c feature/status-command
```

After making changes, run the tests and push the branch:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git status
git add <changed-files>
git commit -m "Add status command"
git push -u origin feature/status-command
```

Open a pull request on GitHub and have at least one teammate review it before
merging. Avoid having multiple people implement the same module simultaneously;
split work by feature and coordinate shared changes to `cli.py`, repository
formats, and configuration schemas.

Never commit `.venv/`, generated `output/` data, `.embeddingvc/` object data,
credentials, or downloaded embedding models. The project `.gitignore` already
excludes local build and virtual-environment files.
