"""List and create local embedding experiment branches."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class BranchError(Exception):
    """A branch operation cannot safely proceed."""


_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_HEX_REVISION = re.compile(r"[0-9a-fA-F]{4,}\Z")


def find_repository(directory: Path | str = ".") -> Path:
    """Find the nearest EmbeddingVC repository at or above *directory*."""
    path = Path(directory).absolute()
    if not path.is_dir():
        raise BranchError(f"Not a directory: {path}")
    for candidate in (path, *path.parents):
        metadata = candidate / ".embeddingvc"
        if metadata.is_dir() and not metadata.is_symlink():
            return candidate
    raise BranchError("Not an EmbeddingVC repository (missing .embeddingvc)")


def _metadata(root: Path) -> Path:
    metadata = root / ".embeddingvc"
    if not metadata.is_dir() or metadata.is_symlink():
        raise BranchError(f"Not an EmbeddingVC repository: {root}")
    return metadata


def _safe_file(path: Path, description: str) -> Path:
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        raise BranchError(f"Linked {description} is not supported: {path.name}")
    if not path.is_file():
        raise BranchError(f"Missing {description}: {path.name}")
    return path


def _head(root: Path) -> tuple[str, str | None]:
    value = _safe_file(root / ".embeddingvc" / "HEAD", "HEAD").read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = value[5:]
        if not ref.startswith("refs/heads/") or "/" in ref[len("refs/heads/"):]:
            raise BranchError("Invalid HEAD reference")
        name = ref[len("refs/heads/"):]
        ref_path = root / ".embeddingvc" / ref
        if not ref_path.is_file() or ref_path.is_symlink():
            raise BranchError(f"HEAD points to a missing branch: {name}")
        commit = ref_path.read_text(encoding="utf-8").strip() or None
        return name, commit
    if not value:
        raise BranchError("HEAD is empty")
    return "", value


def _commit_ids(root: Path) -> list[str]:
    commits = root / ".embeddingvc" / "commits"
    if not commits.is_dir() or commits.is_symlink():
        raise BranchError("Missing commits directory")
    ids: list[str] = []
    for entry in commits.iterdir():
        if entry.is_file() and not entry.is_symlink() and entry.name != ".lock":
            ids.append(entry.name)
    return ids


def _resolve_commit(root: Path, revision: str, ids: list[str] | None = None) -> str:
    if not revision or "/" in revision or "\\" in revision:
        raise BranchError(f"Invalid revision: {revision!r}")
    ids = _commit_ids(root) if ids is None else ids
    exact = [commit for commit in ids if commit == revision]
    if exact:
        return exact[0]
    matches = [commit for commit in ids if commit.lower().startswith(revision.lower())]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise BranchError(f"Ambiguous revision: {revision}")
    raise BranchError(f"Unknown commit: {revision}")


def _short(commit: str | None) -> str:
    return commit[:7] if commit else "(unborn)"


def _branch_paths(root: Path) -> list[Path]:
    heads = root / ".embeddingvc" / "refs" / "heads"
    if not heads.is_dir() or heads.is_symlink():
        raise BranchError("Missing branch refs directory")
    paths = []
    for entry in heads.iterdir():
        if entry.name == ".lock":
            continue
        if entry.is_file() and not entry.is_symlink():
            paths.append(entry)
    return paths


def list_branches(root: Path | str = ".") -> str:
    """Return the human-readable local branch listing."""
    root = find_repository(root)
    current, head_commit = _head(root)
    branches: list[tuple[str, str | None]] = []
    for path in _branch_paths(root):
        branches.append((path.name, path.read_text(encoding="utf-8").strip() or None))
    branches.sort(key=lambda item: item[0])

    lines: list[str] = []
    if current:
        for name, commit in branches:
            marker = "*" if name == current else " "
            lines.append(f"{marker} {name:<20} {_short(commit)}")
        if not any(name == current for name, _ in branches):
            lines.append(f"* {current:<20} {_short(head_commit)}")
    else:
        lines.append(f"* (HEAD detached at {_short(head_commit)})")
        for name, commit in branches:
            lines.append(f"  {name:<20} {_short(commit)}")
    return "\n".join(lines)


def _validate_name(name: str) -> None:
    if not _NAME.fullmatch(name):
        raise BranchError("Invalid branch name; use letters, numbers, '.', '_' or '-' only")
    if name in {"HEAD", ".", ".."} or name.endswith(".lock"):
        raise BranchError(f"Invalid branch name: {name}")
    if _HEX_REVISION.fullmatch(name):
        raise BranchError(f"Branch name looks like a commit revision: {name}")


@contextmanager
def _repository_lock(root: Path) -> Iterator[None]:
    lock = root / ".embeddingvc" / "refs" / "heads" / ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise BranchError("Repository is locked by another branch operation") from exc
    try:
        os.close(fd)
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def create_branch(root: Path | str = ".", name: str = "", start: str | None = None) -> str:
    """Create *name* at a resolved commit and return its full commit id."""
    root = find_repository(root)
    _validate_name(name)
    heads = root / ".embeddingvc" / "refs" / "heads"
    with _repository_lock(root):
        existing = _branch_paths(root)
        comparison = name.casefold() if os.name == "nt" else name
        if any(path.name.casefold() == comparison if os.name == "nt" else path.name == comparison for path in existing):
            raise BranchError(f"Branch already exists: {name}")

        _, current_commit = _head(root)
        revision = start if start is not None else current_commit
        if not revision:
            raise BranchError("Cannot create a branch before the first commit")
        commit = _resolve_commit(root, revision)
        destination = heads / name
        if destination.exists() or destination.is_symlink():
            raise BranchError(f"Branch already exists: {name}")
        try:
            fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(commit + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            try:
                destination.unlink()
            except OSError:
                pass
            raise BranchError(f"Could not create branch {name}: {exc}") from exc
    return commit


def branch(root: Path | str = ".", name: str | None = None, start: str | None = None) -> str:
    """List branches or create one, returning terminal-ready output."""
    if name is None:
        return list_branches(root)
    commit = create_branch(root, name, start)
    return f"Created branch {name} at {_short(commit)}\nTo switch: embeddingvc checkout {name}"
