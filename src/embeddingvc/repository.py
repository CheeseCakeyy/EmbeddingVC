"""Repository initialization without model downloads or database dependencies."""

import json
import os
from importlib.resources import files
from pathlib import Path
from contextlib import contextmanager
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
        ".embeddingvc/config.json": (json.dumps({"repository": {"name": root.name}}, ensure_ascii=True) + "\n").encode(),
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


@contextmanager
def repository_lock(root: Path):
    """Serialize mutations with an OS lock that is released on process death.

    The persistent guard inode must never be removed. The short-lived `lock`
    sentinel remains compatible with older callers; only our own stale sentinel
    is reclaimable while holding the guard, never an unknown/legacy lock.
    """
    from .object_store import reject_links
    from .objects import RepositoryError

    path = root / ".embeddingvc" / "lock"
    guard_path = root / ".embeddingvc" / "mutation.lock"
    reject_links(path, root)
    reject_links(guard_path, root)
    with guard_path.open("a+b") as guard:
        if guard.seek(0, os.SEEK_END) == 0:
            guard.write(b"0")
            guard.flush()
        guard.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RepositoryError("Repository is locked by another operation (.embeddingvc/lock)") from exc
        owned = False
        try:
            token = b"embeddingvc-advisory-v1\n"
            if path.exists():
                if path.read_bytes() != token:
                    raise RepositoryError("Repository is locked by another operation (.embeddingvc/lock)")
                path.unlink()  # The guard proves no current caller owns it.
            with path.open("xb") as sentinel:
                owned = True
                sentinel.write(token)
            from .commit_manager import recover_publication
            from .restore_engine import recover_checkout, TRANSACTION
            checkout_marker = root / TRANSACTION
            reject_links(checkout_marker, root)
            commit_marker = root / ".embeddingvc/transactions/commit.json"
            reject_links(commit_marker, root)
            if checkout_marker.exists() and commit_marker.exists():
                raise RepositoryError("Conflicting commit and checkout transactions; recovery required")
            recover_publication(root)
            recover_checkout(root)
            yield
        finally:
            if owned:
                path.unlink(missing_ok=True)
            guard.seek(0)
            if os.name == "nt":
                msvcrt.locking(guard.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(guard.fileno(), fcntl.LOCK_UN)
