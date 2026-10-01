"""Small committed fixtures: no model, database or source scan required."""
import copy
from embeddingvc.commit_manager import store_commit
from embeddingvc.hashing import hash_payload
from embeddingvc.object_store import write_object
from embeddingvc.objects import write_chunk_object


def snapshot(root, documents=None, parent=None, model="a", message="snapshot"):
    effective = {"dimension": 2, "provenance": {"model": model}}
    config = write_object(root, "configs", effective)
    docs = {}
    for name, entries in (documents or {}).items():
        occurrences = []
        for ordinal, (text, vector) in enumerate(entries):
            chunk = write_chunk_object(root, text)
            embedding = write_object(root, "embeddings", {"schema": 1, "chunk_hash": chunk,
                "embedding_config_hash": config, "dimension": 2, "vector": vector,
                "provenance": effective["provenance"]})
            occurrences.append({"id": hash_payload([name, ordinal]), "ordinal": ordinal,
                "start": 0, "end": len(text), "pages": [], "chunk": chunk,
                "embedding": {"vector": embedding, "embedding_config_hash": config}})
        docs[name] = {"content_hash": hash_payload([text for text, _ in entries]), "occurrences": occurrences}
    manifest = {"version": 1, "parent": parent, "message": message,
        "timestamp": "2026-10-01T00:00:00+00:00", "documents": docs, "tracked_roots": [],
        "config": config, "config_object": config, "embedding_config": config,
        "embedding_config_hash": config, "statistics": {}}
    digest = store_commit(root, manifest)
    (root / ".embeddingvc/refs/heads/main").write_text(digest)
    return digest, copy.deepcopy(manifest)


def state(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}
