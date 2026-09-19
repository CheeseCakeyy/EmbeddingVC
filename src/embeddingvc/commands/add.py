"""`embeddingvc add` - track sources and stage document and chunk state.

Staging is deterministic and side-effect free with respect to models: no
embedding is computed and no vector store is touched. Re-running `add` over
unchanged files reproduces the same chunk objects, the same occurrence IDs and
the same order, so a rescan reports no modifications.

The command is atomic at the index. Every requested file is read, chunked and
hashed into a complete next index held in memory; only once all of them have
succeeded is that index published. A parse failure therefore leaves the
previously staged state exactly as it was. Chunk objects written before a
failure are harmless: they are immutable, content-addressed and unreferenced,
so they are simply never read.

Scope rules. A run refreshes only the paths it was asked about and leaves
every other staged document alone. Within that scope a tracked file that has
disappeared stages a deletion, while a path named on the command line that
exists in neither the working tree nor the index is an error - that is almost
always a typo, and treating it as a deletion would quietly discard state.
"""

from dataclasses import dataclass, field
from pathlib import Path

from .. import config as configuration
from .. import objects
from ..chunking import split_text
from ..document_loader import DocumentError, extractor_versions, load
from ..hashing import hash_bytes, occurrence_id
from ..objects import EXCLUDED_DIRECTORIES, MANAGED_FILES, RepositoryError


class AddError(Exception):
    """Staging cannot proceed; the index is left untouched."""


@dataclass
class Request:
    """One path named on the command line, and what it resolved to."""

    display: str
    relative: str
    is_directory: bool
    files: list[Path] = field(default_factory=list)
    documents: int = 0
    occurrences: int = 0


@dataclass
class Result:
    requests: list[Request]
    added: int = 0
    modified: int = 0
    deleted: int = 0
    unchanged: int = 0
    skipped: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Format the report printed on success."""
        lines = []
        for request in self.requests:
            lines.append(
                f"Staged {request.display}: "
                f"{_plural(request.documents, 'document')}, "
                f"{_plural(request.occurrences, 'chunk occurrence')}"
            )
        lines.append(
            f"Documents: {self.added} added, {self.modified} modified, {self.deleted} deleted"
        )
        if self.skipped:
            shown = ", ".join(self.skipped[:5])
            more = f", and {len(self.skipped) - 5} more" if len(self.skipped) > 5 else ""
            lines.append(f"Skipped {_plural(len(self.skipped), 'unsupported file')}: {shown}{more}")
        lines.append("Next: embeddingvc status")
        return "\n".join(lines)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _reject_links(path: Path, root: Path) -> None:
    """Refuse a path that is, or sits below, a link.

    A link could point anywhere, including outside the repository, so the
    contents it exposes are not safely part of this repository's history.
    """
    current = path
    while True:
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise AddError(f"Linked path is not supported: {current}")
        if current == root or current.parent == current:
            return
        current = current.parent


def _relative(path: Path, root: Path) -> str:
    """Return a repository-relative POSIX path, refusing anything outside."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        raise AddError(f"Path is outside the repository: {path}") from None


def _walk(directory: Path, root: Path, readable: tuple[str, ...], declared: tuple[str, ...],
          files: list[Path], skipped: list[str]) -> None:
    """Collect readable files below a directory, in sorted order.

    Sorting by name at every level makes the traversal order a property of the
    tree rather than of the filesystem, which is what lets two scans of the
    same tree produce identical output.
    """
    try:
        entries = sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise AddError(f"Directory cannot be read: {exc}") from exc
    for entry in entries:
        if entry.is_symlink() or (hasattr(entry, "is_junction") and entry.is_junction()):
            raise AddError(f"Linked path is not supported: {entry}")
        if entry.is_dir():
            if entry.name in EXCLUDED_DIRECTORIES:
                continue
            _walk(entry, root, readable, declared, files, skipped)
        elif entry.is_file():
            if entry.parent == root and entry.name in MANAGED_FILES:
                continue
            suffix = entry.suffix.lower()
            if suffix in readable:
                files.append(entry)
            elif suffix in declared:
                # Configured but unimplemented, e.g. .docx. Worth naming so the
                # user is not left wondering why the file was ignored.
                skipped.append(_relative(entry, root))
            else:
                skipped.append(_relative(entry, root))


def _prune_nested(roots: set[str]) -> list[str]:
    """Drop any tracked root contained in another, keeping the outermost.

    Adding data/ after data/notes.md must not leave both recorded, or the
    document under the inner root would be counted twice.
    """
    ordered = sorted(roots)
    kept = []
    for candidate in ordered:
        if any(candidate != other and _within(candidate, other) for other in ordered):
            continue
        kept.append(candidate)
    return kept


def _within(candidate: str, container: str) -> bool:
    """True when `candidate` is `container` or sits below it."""
    if container == ".":
        return True
    return candidate == container or candidate.startswith(f"{container}/")


def _in_scope(document: str, roots: list[str]) -> bool:
    return any(_within(document, item) for item in roots)


def _resolve(raw: str, root: Path, index: dict, readable: tuple[str, ...],
             declared: tuple[str, ...], skipped: list[str]) -> Request:
    """Turn one command-line path into the set of files it covers."""
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    # Normalize '..' segments textually first: resolve() would follow links and
    # hide an escape that should be reported.
    candidate = Path(candidate.as_posix()) if candidate.as_posix() == str(candidate) else candidate
    try:
        normalized = Path(candidate).absolute()
        normalized = Path(*normalized.parts)
        normalized = normalized.resolve() if not normalized.exists() else normalized.absolute()
    except OSError as exc:
        raise AddError(f"Path cannot be read: {raw}: {exc}") from exc
    normalized = Path(_collapse(normalized))

    relative = _relative(normalized, root)
    if relative != "." and any(part in EXCLUDED_DIRECTORIES for part in Path(relative).parts):
        raise AddError(f"Path is managed by EmbeddingVC and cannot be tracked: {raw}")

    if normalized.exists():
        _reject_links(normalized, root)

    if normalized.is_dir():
        files: list[Path] = []
        _walk(normalized, root, readable, declared, files, skipped)
        return Request(display=f"{relative}/" if relative != "." else "./",
                       relative=relative, is_directory=True, files=files)

    if normalized.is_file():
        if normalized.parent == root and normalized.name in MANAGED_FILES:
            raise AddError(
                f"'{relative}' is generated by EmbeddingVC and describes the repository, so it is not tracked as a document."
            )
        suffix = normalized.suffix.lower()
        if suffix not in readable:
            raise AddError(
                f"'{suffix}' documents cannot be read yet: {raw}. "
                f"Readable types are {', '.join(readable) or 'none'}."
            )
        return Request(display=relative, relative=relative, is_directory=False, files=[normalized])

    # Missing. Tracked files stage a deletion; anything else is a mistake.
    tracked = index["documents"]
    if relative in tracked:
        return Request(display=relative, relative=relative, is_directory=False, files=[])
    if any(_within(document, relative) for document in tracked):
        return Request(display=f"{relative}/", relative=relative, is_directory=True, files=[])
    raise AddError(f"Path does not exist: {raw}")


def _collapse(path: Path) -> str:
    """Remove '.' and '..' segments without touching the filesystem."""
    parts: list[str] = []
    for part in path.parts:
        if part == ".":
            continue
        if part == ".." and parts and parts[-1] not in ("..", path.anchor):
            parts.pop()
            continue
        parts.append(part)
    return str(Path(*parts)) if parts else str(path)


def _stage_document(root: Path, path: Path, relative: str, settings: configuration.Config,
                    previous: dict | None, reuse_vectors: bool) -> dict:
    """Read, chunk and store one document, returning its index entry."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AddError(f"{relative}: cannot be read: {exc}") from exc
    content = hash_bytes(raw)

    if previous is not None and reuse_vectors and previous.get("content_hash") == content:
        # Same bytes under the same staging snapshot: the stored occurrences
        # are by definition what this run would recompute.
        return previous

    try:
        document = load(path, settings)
    except DocumentError as exc:
        raise AddError(str(exc)) from exc

    carried = {}
    if previous is not None and reuse_vectors:
        fingerprint = settings.embedding_fingerprint()
        for occurrence in previous.get("occurrences", []):
            embedding = occurrence.get("embedding")
            if embedding and embedding.get("fingerprint") == fingerprint:
                carried.setdefault(occurrence["chunk"], embedding)

    occurrences = []
    chunks = split_text(document.text, settings.chunking.chunk_size, settings.chunking.chunk_overlap)
    for ordinal, chunk in enumerate(chunks):
        digest = objects.write_chunk_object(root, chunk.text)
        occurrences.append({
            "id": occurrence_id(relative, ordinal),
            "chunk": digest,
            "ordinal": ordinal,
            "start": chunk.start,
            "end": chunk.end,
            "pages": document.pages_for(chunk.start, chunk.end),
            # A vector is carried forward only when the identical chunk text was
            # already embedded under identical embedding settings.
            "embedding": carried.get(digest),
        })
    return {
        "content_hash": content,
        "size": len(raw),
        "extractor": document.extractor,
        "occurrences": occurrences,
    }


def add(paths: list[str], *, directory: Path | str = ".") -> Result:
    """Stage the given files and directories. Returns a summary on success."""
    if not paths:
        raise AddError("Nothing to add. Name at least one file or directory.")
    try:
        root = objects.find_repository(Path(directory))
        index = objects.read_index(root)
    except RepositoryError as exc:
        raise AddError(str(exc)) from exc
    try:
        settings = configuration.load(root)
    except configuration.ConfigError as exc:
        raise AddError(str(exc)) from exc

    readable = settings.readable_extensions
    if not readable:
        raise AddError(
            "No readable document types are configured. "
            f"Set 'documents.supported_types' to include one of {', '.join(configuration.READABLE_EXTENSIONS)}."
        )

    skipped: list[str] = []
    requests = [_resolve(raw, root, index, readable, settings.declared_extensions, skipped) for raw in paths]

    try:
        snapshot = settings.staging_snapshot(extractor_versions(readable))
    except DocumentError as exc:
        raise AddError(str(exc)) from exc
    config_digest = objects.write_config_object(root, snapshot)
    # Staged chunks may only be reused while the snapshot that produced them is
    # still in force; otherwise every tracked document is re-chunked.
    reuse = index.get("config") == config_digest

    scope = [request.relative for request in requests]
    documents = dict(index["documents"])
    result = Result(requests=requests)

    seen: set[str] = set()
    for request in requests:
        for path in request.files:
            relative = _relative(path, root)
            if relative in seen:
                continue  # Nested roots must not stage the same file twice.
            seen.add(relative)
            previous = index["documents"].get(relative)
            entry = _stage_document(root, path, relative, settings, previous, reuse)
            if previous is None:
                result.added += 1
            elif previous.get("content_hash") != entry["content_hash"] or not reuse:
                if previous.get("content_hash") == entry["content_hash"]:
                    result.unchanged += 1
                else:
                    result.modified += 1
            else:
                result.unchanged += 1
            documents[relative] = entry
            request.documents += 1
            request.occurrences += len(entry["occurrences"])

    for relative in sorted(index["documents"]):
        if relative in seen or not _in_scope(relative, scope):
            continue
        if not (root / relative).is_file():
            del documents[relative]
            result.deleted += 1

    roots = set(index.get("tracked_roots", [])) | {request.relative for request in requests}
    # A root whose documents have all gone stops being tracked, so a later run
    # does not keep reporting a directory that no longer exists.
    roots = {item for item in roots if (root / item).exists() or any(_within(name, item) for name in documents)}

    index["documents"] = documents
    index["tracked_roots"] = _prune_nested(roots)
    index["config"] = config_digest
    try:
        objects.publish_index(root, index)
    except OSError as exc:
        raise AddError(f"Index cannot be published: {exc}") from exc

    result.skipped = sorted(set(skipped))
    return result
