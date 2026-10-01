"""Chroma projection using explicit vectors and verified replacement collections."""

import math
import uuid
from pathlib import Path

from ..hashing import hash_payload
from ..object_store import reject_links, validate_vector
from ..objects import RepositoryError


class ChromaAdapter:
    def __init__(self, root, settings, *, client=None):
        path = Path(settings["persist_directory"])
        path = path if path.is_absolute() else root / path
        reject_links(path, root)
        path = path.resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise RepositoryError("Vector store path must stay inside repository") from exc
        if path == root or (root / ".embeddingvc") in (path, *path.parents):
            raise RepositoryError("Vector store must not write to repository metadata")
        self.prefix = "evc-" + hash_payload(settings["collection"])[:12]
        if client is None:
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError as exc:
                raise RepositoryError('Install Chroma with: pip install -e ".[vector-store]"') from exc
            client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        self.client = client

    def replace(self, commit, records, dimension):
        expected = {row["id"]: row for row in records}
        if len(expected) != len(records):
            raise RepositoryError("Duplicate vector store occurrence IDs")
        for row in records:
            validate_vector(row["vector"], dimension)
        name = f"{self.prefix}-{commit[:12]}-{uuid.uuid4().hex}"
        collection = self.client.create_collection(name=name, embedding_function=None,
                                                  metadata={"commit": commit, "dimension": dimension})
        batch_size = min(1000, self.client.get_max_batch_size())
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            collection.add(ids=[row["id"] for row in batch],
                           embeddings=[row["vector"] for row in batch],
                           documents=[row["text"] for row in batch],
                           metadatas=[row["metadata"] for row in batch])
        if collection.count() != len(records):
            raise RepositoryError("Chroma membership count mismatch")
        seen = set()
        for offset in range(0, len(records), batch_size):
            page = collection.get(limit=batch_size, offset=offset,
                                  include=["embeddings", "documents", "metadatas"])
            for i, identifier in enumerate(page["ids"]):
                if identifier not in expected or identifier in seen:
                    raise RepositoryError("Chroma membership mismatch")
                seen.add(identifier)
                row = expected[identifier]
                values = page["embeddings"][i]
                values = values.tolist() if hasattr(values, "tolist") else values
                values = validate_vector(values, dimension)
                if (any(not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-7)
                        for a, b in zip(values, row["vector"]))
                        or page["documents"][i] != row["text"] or page["metadatas"][i] != row["metadata"]):
                    raise RepositoryError("Chroma record verification failed")
        if seen != set(expected):
            raise RepositoryError("Chroma membership mismatch")
        return name
