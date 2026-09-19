"""Reading document text, with page provenance and honest failures.

A load either succeeds completely or raises `DocumentError`. Nothing is
reported as empty because it could not be parsed: the caller needs that
distinction, because an unreadable file must abort staging rather than be
recorded as a document with no chunks.

Text is normalized per page and then joined, and the spans returned describe
the *normalized* text. Normalizing first and recording spans afterwards keeps
chunk offsets valid; normalizing a joined string would invalidate any offsets
collected before it.
"""

from dataclasses import dataclass
from pathlib import Path

from .config import Config

TEXT_EXTENSIONS = (".txt", ".md")
PDF_EXTENSION = ".pdf"

#: Inserted between pages so a chunk cannot silently run the last word of one
#: page into the first word of the next. Counts towards chunk size.
PAGE_SEPARATOR = "\n\n"


class DocumentError(Exception):
    """A document exists but its text cannot be extracted."""


@dataclass(frozen=True)
class Span:
    """A half-open range of the normalized text belonging to one page."""

    start: int
    end: int
    page: int | None


@dataclass(frozen=True)
class LoadedDocument:
    text: str
    spans: tuple[Span, ...]
    extractor: str
    extractor_version: str

    def pages_for(self, start: int, end: int) -> list[int]:
        """Return the page numbers a half-open character range touches."""
        pages = []
        for span in self.spans:
            if span.page is None:
                continue
            if span.start < end and start < span.end:
                pages.append(span.page)
        return sorted(set(pages))


def pymupdf_version() -> str:
    """Return the pinned PyMuPDF version, for the staging snapshot."""
    try:
        import pymupdf
    except ModuleNotFoundError:
        try:
            import fitz as pymupdf
        except ModuleNotFoundError as exc:
            raise DocumentError(
                "PyMuPDF is required to read PDF documents. Install the project dependencies."
            ) from exc
    version = getattr(pymupdf, "VersionBind", None)
    if not version:
        version = getattr(pymupdf, "version", ("unknown",))[0]
    return str(version)


def extractor_versions(extensions: tuple[str, ...]) -> dict:
    """Report the version of each extractor the configured types will use.

    Only extractors that can actually be reached are reported, so a repository
    holding no PDFs does not invalidate its staging when PyMuPDF is upgraded.
    """
    versions = {}
    if any(item in TEXT_EXTENSIONS for item in extensions):
        versions["text"] = "1"
    if PDF_EXTENSION in extensions:
        versions["pdf"] = f"pymupdf-{pymupdf_version()}"
    return versions


def _load_text(path: Path, config: Config) -> LoadedDocument:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DocumentError(f"{path}: cannot be read: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentError(
            f"{path}: is not valid UTF-8 (byte {exc.start}). Re-save it as UTF-8 to stage it."
        ) from exc
    normalized = config.preprocessing.apply(text)
    spans = (Span(0, len(normalized), None),) if normalized else ()
    return LoadedDocument(text=normalized, spans=spans, extractor="text", extractor_version="1")


def _load_pdf(path: Path, config: Config) -> LoadedDocument:
    try:
        import pymupdf
    except ModuleNotFoundError:
        try:
            import fitz as pymupdf
        except ModuleNotFoundError as exc:
            raise DocumentError(
                f"{path}: PyMuPDF is required to read PDF documents. Install the project dependencies."
            ) from exc
    try:
        document = pymupdf.open(path)
    except Exception as exc:  # PyMuPDF raises library-specific errors
        raise DocumentError(f"{path}: is not a readable PDF: {exc}") from exc
    try:
        if document.needs_pass:
            raise DocumentError(f"{path}: is encrypted. Remove the password to stage it.")
        pieces: list[str] = []
        spans: list[Span] = []
        carries_graphics = False
        cursor = 0
        for number, page in enumerate(document, start=1):
            try:
                raw = page.get_text()
            except Exception as exc:
                raise DocumentError(f"{path}: page {number} cannot be extracted: {exc}") from exc
            if not carries_graphics and not raw.strip():
                # Remembered so a wholly text-free PDF can be told apart from a
                # scanned one: blank pages are legitimate, images need OCR.
                try:
                    carries_graphics = bool(page.get_images())
                except Exception:
                    carries_graphics = False
            normalized = config.preprocessing.apply(raw)
            if not normalized:
                continue
            if pieces:
                cursor += len(PAGE_SEPARATOR)
            pieces.append(normalized)
            spans.append(Span(cursor, cursor + len(normalized), number))
            cursor += len(normalized)
        if not pieces and carries_graphics:
            raise DocumentError(
                f"{path}: holds images but no extractable text. OCR is not supported yet, so it cannot be staged."
            )
        return LoadedDocument(
            text=PAGE_SEPARATOR.join(pieces),
            spans=tuple(spans),
            extractor="pdf",
            extractor_version=f"pymupdf-{pymupdf_version()}",
        )
    finally:
        document.close()


def load(path: Path, config: Config) -> LoadedDocument:
    """Read one document, dispatching on its extension."""
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return _load_text(path, config)
    if suffix == PDF_EXTENSION:
        return _load_pdf(path, config)
    raise DocumentError(f"{path}: '{suffix}' documents cannot be read yet.")
