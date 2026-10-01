"""Refresh tracked sources and atomically publish a fully embedded candidate."""

import copy
import time
from dataclasses import dataclass
from pathlib import Path

from .add import AddError, _prune_nested, _resolve, _stage_document
from .. import config, objects, staging_config
from ..document_loader import DocumentError, extractor_versions
from ..embedding_engine import (
    EmbeddingError, SentenceTransformerEncoder, effective_configuration, encode_batch, validate_revision,
)
from ..hashing import hash_bytes, hash_payload
from ..object_store import (
    find_embeddings, read_object, reject_links, valid_digest, write_object,
)
from ..repository import repository_lock


class EmbedError(Exception):
    """Generation failed; the previous candidate index remains authoritative."""


@dataclass(frozen=True)
class Result:
    generated: int
    reused: int
    occurrences: int
    elapsed_seconds: float

    def render(self) -> str:
        return (f"Embedding complete: {self.generated} new objects, {self.reused} reused occurrences\n"
                f"{self.occurrences} chunk occurrences ready\n"
                f"Elapsed: {self.elapsed_seconds:.2f}s\n"
                'Next: embeddingvc commit -m "Describe this update"')


def _files(root, roots, index, settings):
    files, skipped = {}, []
    for relative in roots:
        if (not isinstance(relative, str) or not relative or "\\" in relative
                or ":" in relative or Path(relative).is_absolute() or ".." in Path(relative).parts):
            raise EmbedError("Invalid tracked root in index")
        path = root / relative
        reject_links(path, root)
        if not path.exists():
            continue  # A deleted tracked root is a legitimate empty scope.
        request = _resolve(str(path), root, index, settings.readable_extensions,
                           settings.declared_extensions, skipped)
        for file in request.files:
            files[file.relative_to(root).as_posix()] = file
    return dict(sorted(files.items()))


def _validate_index(index):
    for relative in [*index["tracked_roots"], *index["documents"]]:
        if (not isinstance(relative, str) or not relative or "\\" in relative
                or ":" in relative or Path(relative).is_absolute() or ".." in Path(relative).parts):
            raise EmbedError("Invalid tracked path in index")
    for name, document in index["documents"].items():
        if (not isinstance(document, dict) or not valid_digest(document.get("content_hash"))
                or not isinstance(document.get("occurrences"), list)):
            raise EmbedError(f"Invalid document record: {name}")
        for occurrence in document["occurrences"]:
            if not isinstance(occurrence, dict) or not valid_digest(occurrence.get("chunk")):
                raise EmbedError(f"Invalid chunk occurrence: {name}")


def embed(directory: Path | str = ".", *, encoder_factory=None) -> Result:
    """Generate once per compatible unique chunk, retaining historical objects.

    encoder_factory is a test seam; CLI callers always use Sentence Transformers.
    """
    started = time.perf_counter()
    try:
        root = objects.find_repository(Path(directory))
        with repository_lock(root):
            return _embed(root, started, encoder_factory or SentenceTransformerEncoder)
    except (objects.RepositoryError, config.ConfigurationError, staging_config.ConfigError,
            AddError, DocumentError, EmbeddingError, OSError, ValueError, TypeError) as exc:
        raise EmbedError(str(exc)) from exc


def _embed(root, started, encoder_factory):
    index_path = objects.index_path(root)
    reject_links(index_path, root)
    previous_bytes = index_path.read_bytes()
    index = objects.read_index(root)
    _validate_index(index)
    roots = _prune_nested(set(index["tracked_roots"] or index["documents"]))
    if not roots:
        raise EmbedError("No tracked inputs. Run embeddingvc add data first.")
    configuration_path = root / "embeddingvc.yaml"
    reject_links(configuration_path, root)
    configuration_bytes = configuration_path.read_bytes()
    complete_config = config.validate(config.parse_yaml(configuration_bytes.decode("utf-8")), root).data
    settings = staging_config.parse(config.dump_yaml(complete_config))
    validate_revision(settings.embedding)
    staging = settings.staging_snapshot(extractor_versions(settings.readable_extensions))
    staging_hash = write_object(root, "configs", staging)
    files = _files(root, roots, index, settings)

    candidate = copy.deepcopy(index)
    candidate.setdefault("base_commit", None)
    candidate["tracked_roots"] = roots
    candidate["config"] = staging_hash
    candidate["config_object"] = write_object(root, "configs", complete_config)
    candidate["documents"] = {}
    for relative, file in files.items():
        # Re-read every source; no previous occurrence can bypass validation.
        entry = _stage_document(root, file, relative, settings, None, False)
        entry["config"] = staging_hash
        candidate["documents"][relative] = entry

    encoder = encoder_factory(settings.embedding)
    expected = settings.embedding.get("dimension")
    dimension = encoder.dimension
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise EmbedError("Model did not report a positive embedding dimension")
    if expected is not None and dimension != expected:
        raise EmbedError(f"Model dimension mismatch: configured {expected}, model reports {dimension}")
    provenance = encoder.provenance
    if not isinstance(provenance, dict) or not provenance:
        raise EmbedError("Model/tokenizer provenance is missing")
    effective = effective_configuration(settings, staging, dimension, provenance)
    config_hash = hash_payload(effective)
    candidate["embedding_config_hash"] = config_hash
    candidate["embedding_config"] = write_object(root, "configs", effective)
    reusable = find_embeddings(root, config_hash, dimension, provenance)

    chunks = {}
    for document in candidate["documents"].values():
        for occurrence in document["occurrences"]:
            digest = occurrence["chunk"]
            chunks.setdefault(digest, read_object(root, "chunks", digest)["text"])
    # A referenced object must not silently disappear or be replaced on retry.
    for document in index["documents"].values():
        for occurrence in document.get("occurrences", []):
            reference = occurrence.get("embedding")
            if isinstance(reference, dict) and reference.get("embedding_config_hash") == config_hash:
                payload = read_object(root, "embeddings", reference.get("vector"))
                if payload["chunk_hash"] != occurrence["chunk"] or payload["embedding_config_hash"] != config_hash:
                    raise EmbedError("Embedding reference does not match its chunk/configuration")

    pending = sorted(digest for digest in chunks if digest not in reusable)
    resolved = dict(reusable)
    batch_size = settings.embedding["batch_size"]
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        vectors = encode_batch(encoder, [chunks[digest] for digest in batch], batch_size=batch_size,
                               normalize_embeddings=settings.embedding["normalize_embeddings"])
        for digest, vector in zip(batch, vectors):
            resolved[digest] = write_object(root, "embeddings", {
                "schema": 1, "chunk_hash": digest, "embedding_config_hash": config_hash,
                "vector": vector, "dimension": dimension, "provenance": provenance,
            })

    count = reused = 0
    assigned_new = set()
    for document in candidate["documents"].values():
        for occurrence in document["occurrences"]:
            digest = occurrence["chunk"]
            occurrence["embedding"] = {"vector": resolved[digest],
                                       "fingerprint": settings.embedding_fingerprint(),
                                       "embedding_config_hash": config_hash}
            count += 1
            reused += digest in reusable or digest in assigned_new
            assigned_new.add(digest)
            # Verify again, including objects just written, before publication.
            read_object(root, "chunks", digest)
            payload = read_object(root, "embeddings", resolved[digest])
            if (payload["chunk_hash"] != digest or payload["embedding_config_hash"] != config_hash
                    or payload["dimension"] != dimension or payload["provenance"] != provenance):
                raise EmbedError("Embedding object changed during generation")
    for key in ("config", "config_object", "embedding_config"):
        read_object(root, "configs", candidate[key])

    # Detect edits, new files, deletions, config changes and external staging
    # while a slow model was running. Only this final replace publishes state.
    current_files = _files(root, roots, index, settings)
    fingerprints = {name: hash_bytes(file.read_bytes()) for name, file in current_files.items()}
    expected_sources = {name: doc["content_hash"] for name, doc in candidate["documents"].items()}
    current_staging = settings.staging_snapshot(extractor_versions(settings.readable_extensions))
    if (fingerprints != expected_sources or configuration_path.read_bytes() != configuration_bytes
            or current_staging != staging or index_path.read_bytes() != previous_bytes):
        raise EmbedError("Sources, configuration or index changed during embedding. Retry embeddingvc embed.")
    result = Result(len(pending), reused, count, time.perf_counter() - started)
    candidate["generation"] = {"status": "ready", "new_objects": result.generated,
                               "reused_occurrences": reused, "occurrences": count,
                               "elapsed_seconds": result.elapsed_seconds}
    objects.publish_index(root, candidate)
    return result
