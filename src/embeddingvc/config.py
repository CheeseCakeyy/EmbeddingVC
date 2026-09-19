"""Reading and validating embeddingvc.yaml.

PROVISIONAL CONTRACT. `add` is the first command to parse the configuration
template that `init` writes, so the field names, the permitted values and the
shape of the snapshot below are proposals awaiting the embed owner's review.

Two things are produced from one file:

* a `Config` - the validated settings the staging pipeline reads.
* a *staging snapshot* - the subset of settings that changes staged output,
  plus the versions of the extractors used. It is stored as an object and
  recorded in the index. When the snapshot hash changes, previously staged
  chunks can no longer be reused and every tracked document is re-chunked.

The snapshot deliberately excludes settings that do not affect chunk text, so
renaming a collection or changing a batch size does not invalidate staging.
"""

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .hashing import hash_payload

CONFIG_FILENAME = "embeddingvc.yaml"

NORMALIZATION_FORMS = ("NFC", "NFD", "NFKC", "NFKD")
CHUNKING_STRATEGIES = ("recursive-character",)

#: Extensions the loader can read today. DOCX is listed in the template's
#: supported_types but is not implemented yet, so it is skipped with a notice
#: rather than silently treated as missing.
READABLE_EXTENSIONS = (".txt", ".md", ".pdf")


class ConfigError(Exception):
    """The configuration file is missing, malformed, or holds an unusable value."""


@dataclass(frozen=True)
class Preprocessing:
    unicode_normalization: str
    remove_extra_whitespace: bool
    lowercase: bool

    def apply(self, text: str) -> str:
        """Normalize one page or file of text.

        Whitespace collapsing runs after Unicode normalization because NFKC
        rewrites several exotic spaces into U+0020, and the result should then
        collapse with its neighbours rather than survive as a double space.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = unicodedata.normalize(self.unicode_normalization, text)
        if self.remove_extra_whitespace:
            lines = [" ".join(line.split()) for line in text.split("\n")]
            text = "\n".join(lines)
            while "\n\n\n" in text:
                text = text.replace("\n\n\n", "\n\n")
            text = text.strip()
        if self.lowercase:
            text = text.lower()
        return text


@dataclass(frozen=True)
class Chunking:
    strategy: str
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class Config:
    source_directory: str
    supported_types: tuple[str, ...]
    preprocessing: Preprocessing
    chunking: Chunking
    embedding: dict = field(default_factory=dict)

    @property
    def readable_extensions(self) -> tuple[str, ...]:
        """Configured types intersected with what the loader actually supports."""
        configured = tuple(f".{item}" for item in self.supported_types)
        return tuple(item for item in configured if item in READABLE_EXTENSIONS)

    @property
    def declared_extensions(self) -> tuple[str, ...]:
        """Every configured type, including ones no loader handles yet."""
        return tuple(f".{item}" for item in self.supported_types)

    def staging_snapshot(self, extractors: dict) -> dict:
        """Return the settings that determine staged chunk text.

        `extractors` maps a loader name to its version string; a PyMuPDF
        upgrade that changes text extraction must invalidate staging, so the
        versions belong inside the hashed snapshot rather than beside it.
        """
        return {
            "schema": 1,
            "tool_version": __version__,
            "preprocessing": {
                "unicode_normalization": self.preprocessing.unicode_normalization,
                "remove_extra_whitespace": self.preprocessing.remove_extra_whitespace,
                "lowercase": self.preprocessing.lowercase,
            },
            "chunking": {
                "strategy": self.chunking.strategy,
                "chunk_size": self.chunking.chunk_size,
                "chunk_overlap": self.chunking.chunk_overlap,
            },
            "extractors": dict(sorted(extractors.items())),
        }

    def embedding_fingerprint(self) -> str:
        """Identify the embedding settings a stored vector would have to match.

        `add` never calls a model, but it must decide whether an existing
        vector reference may be carried forward. A vector stays valid only
        while the model, its pinned revision and the normalization flag are
        unchanged; batch size does not affect the vector produced.
        """
        return hash_payload({
            "model": self.embedding.get("model"),
            "revision": self.embedding.get("revision"),
            "normalize_embeddings": self.embedding.get("normalize_embeddings"),
        })


def _require_mapping(value: object, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"Section '{name}' must be a mapping, found {type(value).__name__}.")
    return value


def _require_bool(section: dict, name: str, key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"'{name}.{key}' must be true or false, found {value!r}.")
    return value


def _require_positive_int(section: dict, name: str, key: str) -> int:
    value = section.get(key)
    # bool is a subclass of int; accepting it here would silently mean 0 or 1.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{name}.{key}' must be a positive whole number, found {value!r}.")
    return value


def parse(text: str) -> Config:
    """Validate configuration text and return the settings `add` depends on."""
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
        raise ConfigError("PyYAML is required to read embeddingvc.yaml. Install the project dependencies.") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Configuration is not valid YAML: {exc}") from exc
    raw = _require_mapping(raw, "<root>")

    documents = _require_mapping(raw.get("documents"), "documents")
    source_directory = documents.get("source_directory", "./data")
    if not isinstance(source_directory, str) or not source_directory.strip():
        raise ConfigError(f"'documents.source_directory' must be a path, found {source_directory!r}.")
    declared = documents.get("supported_types", list(item.lstrip(".") for item in READABLE_EXTENSIONS))
    if not isinstance(declared, list) or not declared:
        raise ConfigError(f"'documents.supported_types' must be a non-empty list, found {declared!r}.")
    types = []
    for item in declared:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"'documents.supported_types' entries must be extensions, found {item!r}.")
        types.append(item.strip().lstrip(".").lower())
    # Sorted and de-duplicated so two configs listing the same types in a
    # different order produce the same behaviour.
    supported_types = tuple(sorted(set(types)))

    pre = _require_mapping(raw.get("preprocessing"), "preprocessing")
    form = pre.get("unicode_normalization", "NFKC")
    if form not in NORMALIZATION_FORMS:
        raise ConfigError(f"'preprocessing.unicode_normalization' must be one of {', '.join(NORMALIZATION_FORMS)}, found {form!r}.")
    preprocessing = Preprocessing(
        unicode_normalization=form,
        remove_extra_whitespace=_require_bool(pre, "preprocessing", "remove_extra_whitespace", True),
        lowercase=_require_bool(pre, "preprocessing", "lowercase", False),
    )

    chunk = _require_mapping(raw.get("chunking"), "chunking")
    strategy = chunk.get("strategy", "recursive-character")
    if strategy not in CHUNKING_STRATEGIES:
        raise ConfigError(f"'chunking.strategy' must be one of {', '.join(CHUNKING_STRATEGIES)}, found {strategy!r}.")
    chunk_size = _require_positive_int(chunk, "chunking", "chunk_size")
    overlap = chunk.get("chunk_overlap", 0)
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise ConfigError(f"'chunking.chunk_overlap' must be zero or a positive whole number, found {overlap!r}.")
    if overlap >= chunk_size:
        # Equal values would make every chunk reproduce its predecessor and the
        # merge loop would never advance.
        raise ConfigError(f"'chunking.chunk_overlap' ({overlap}) must be smaller than 'chunking.chunk_size' ({chunk_size}).")

    return Config(
        source_directory=source_directory,
        supported_types=supported_types,
        preprocessing=preprocessing,
        chunking=Chunking(strategy=strategy, chunk_size=chunk_size, chunk_overlap=overlap),
        embedding=_require_mapping(raw.get("embedding"), "embedding"),
    )


def load(root: Path) -> Config:
    """Read and validate the configuration stored at a repository root."""
    path = root / CONFIG_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"Configuration is not valid UTF-8: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"Configuration cannot be read: {exc}") from exc
    return parse(text)
