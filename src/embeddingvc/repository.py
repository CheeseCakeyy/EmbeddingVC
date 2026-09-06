"""Repository initialization without model downloads or database dependencies."""

import json
from importlib.resources import files
from pathlib import Path
from string import Template


class InitializationError(Exception):
    """Initialization cannot safely proceed."""


IGNORE_RULES = ("/output/", "/.embeddingvc/cache/", "/.embeddingvc/objects/")
DIRECTORIES = ("data", "output", "output/chroma", ".embeddingvc",
               ".embeddingvc/objects", ".embeddingvc/commits", ".embeddingvc/refs",
               ".embeddingvc/refs/heads", ".embeddingvc/cache")


def initialize(directory: Path | str = ".", *, force: bool = False) -> Path:
    """Create an empty repository, restoring touched files on ordinary failures.

    An empty refs/heads/main denotes an unborn branch; HEAD is symbolic.
    Force replaces only config and README, never an existing repository.
    """
    root = Path(directory).absolute()
    # Reject links in the target ancestry and managed paths before resolving them.
    for path in (root, *root.parents):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise InitializationError(f"Linked directory is not supported: {path}")
    if root.exists() and not root.is_dir():
        raise InitializationError(f"Not a directory: {root}")
    metadata = root / ".embeddingvc"
    if metadata.exists() or metadata.is_symlink():
        raise InitializationError(f"Repository already initialized (or metadata exists): {metadata}")
    for relative in (*DIRECTORIES, "README.md", "embeddingvc.yaml", ".gitignore"):
        path = root / relative
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise InitializationError(f"Managed path cannot be a link: {path}")
        if path.exists():
            if relative in DIRECTORIES and not path.is_dir():
                raise InitializationError(f"Required directory is a file: {path}")
            if relative not in DIRECTORIES and not path.is_file():
                raise InitializationError(f"Required file is a directory: {path}")
            if relative in ("README.md", "embeddingvc.yaml") and not force:
                raise InitializationError(f"File already exists: {path}. Use --force to replace it.")

    templates = files("embeddingvc").joinpath("templates")
    config = Template(templates.joinpath("embeddingvc.yaml").read_text(encoding="utf-8")).substitute(
        repository_name=json.dumps(root.name, ensure_ascii=True))
    ignore_path = root / ".gitignore"
    original_ignore = ignore_path.read_bytes() if ignore_path.exists() else b""
    existing_rules = original_ignore.decode("utf-8").splitlines()
    missing = [rule for rule in IGNORE_RULES if rule not in existing_rules]
    ignore = original_ignore
    if missing:
        if ignore and not ignore.endswith(b"\n"):
            ignore += b"\n"
        ignore += ("\n# EmbeddingVC generated artifacts\n" + "\n".join(missing) + "\n").encode()
    contents = {
        "embeddingvc.yaml": config.encode(),
        "README.md": templates.joinpath("README.md").read_bytes(),
        ".embeddingvc/HEAD": b"ref: refs/heads/main\n",
        ".embeddingvc/refs/heads/main": b"",
        ".embeddingvc/index.json": b'{"version": 1, "documents": {}}\n',
    }
    if missing:
        contents[".gitignore"] = ignore
    created_dirs: list[Path] = []
    touched: list[tuple[Path, bytes | None]] = []

    def mkdir(path: Path) -> None:
        if not path.exists():
            mkdir(path.parent)
            path.mkdir()
            created_dirs.append(path)

    try:
        mkdir(root)
        for relative in DIRECTORIES:
            mkdir(root / relative)
        for relative, content in contents.items():
            path = root / relative
            previous = path.read_bytes() if path.exists() else None
            # Exclusive creation avoids overwriting a file created after preflight.
            mode = "xb" if previous is None else "wb"
            with path.open(mode) as stream:
                touched.append((path, previous))
                stream.write(content)
    except OSError as exc:
        cleanup_errors = []
        for path, previous in reversed(touched):
            try:
                if previous is None:
                    path.unlink()
                else:
                    path.write_bytes(previous)
            except OSError as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))
        for path in reversed(created_dirs):
            try:
                path.rmdir()  # Never recursively delete user content.
            except OSError as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))
        detail = f" Rollback incomplete: {'; '.join(cleanup_errors)}" if cleanup_errors else " Changes rolled back."
        raise InitializationError(f"Initialization failed: {exc}.{detail}") from exc
    return root
