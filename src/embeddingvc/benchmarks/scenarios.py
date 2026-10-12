"""Seeded change scenarios applied to a copy of the baseline workspace.

Selections depend only on (seed, scenario), so every repeat and every strategy
sees identical input. Counts round half up and never fall below one document;
the actual count is recorded with each scenario.
"""

import math
import os
import random
from pathlib import Path

from .. import config
from .common import BenchmarkError
from .generate_dataset import TOPICS, Document, document, rewrite_paragraph

SCENARIOS = ("edit-1pct", "edit-5pct", "edit-10pct", "add-5pct", "delete-5pct", "rename-5pct",
             "metadata-only", "chunk-size", "chunk-overlap", "preprocessing", "model-change")
CONFIG_SCENARIOS = frozenset({"chunk-size", "chunk-overlap", "preprocessing", "model-change"})
MODEL_KEYS = ("model", "revision", "pooling", "dimension", "normalize_embeddings")


def changed_count(percent: float, total: int) -> int:
    return max(1, math.floor(total * percent / 100 + 0.5))


def load_model_change(path: str | Path | None) -> dict | None:
    """Read the second, explicitly pinned embedding configuration."""
    if path is None:
        return None
    import yaml
    from ..embedding_engine import EmbeddingError, validate_revision
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BenchmarkError(f"Cannot read --model-change-config: {exc}") from exc
    embedding = raw.get("embedding", raw) if isinstance(raw, dict) else None
    if not isinstance(embedding, dict) or not embedding.get("model"):
        raise BenchmarkError("--model-change-config needs an embedding mapping with model and revision")
    unknown = set(embedding) - set(MODEL_KEYS)
    if unknown:
        raise BenchmarkError(f"Unsupported model-change keys: {', '.join(sorted(unknown))}")
    try:
        validate_revision(embedding)
    except EmbeddingError as exc:
        raise BenchmarkError(f"--model-change-config: {exc}") from exc
    return embedding


def _select(seed: int, scenario: str, documents: list[Document], percent: float) -> list[Document]:
    rng = random.Random(f"{seed}:{scenario}:select")
    return sorted(rng.sample(documents, changed_count(percent, len(documents))), key=lambda doc: doc.name)


def _set(workspace: Path, key: str, value) -> None:
    text = "null" if value is None else str(value).lower() if isinstance(value, bool) else str(value)
    config.set_value(workspace, key, text)


def apply(name: str, workspace: Path, documents: list[Document], seed: int,
          model_change: dict | None) -> tuple[list[Document], dict]:
    """Apply one scenario in place; return the resulting corpus and what changed."""
    data = workspace / "data"
    after = list(documents)
    details = {"scenario": name, "documents_before": len(documents)}
    if name.startswith("edit-"):
        selected = _select(seed, name, documents, float(name[5:-3]))
        replacements = {doc.name: rewrite_paragraph(seed, name, doc) for doc in selected}
        for doc in replacements.values():
            (data / doc.name).write_text(doc.text, encoding="utf-8", newline="\n")
        after = [replacements.get(doc.name, doc) for doc in documents]
        details["changed_documents"] = sorted(replacements)
    elif name == "add-5pct":
        topics = sorted(TOPICS)
        start = max(int(doc.name[4:9]) for doc in documents) + 1
        added = [document(seed, number, topics[number % len(topics)])
                 for number in range(start, start + changed_count(5, len(documents)))]
        for doc in added:
            (data / doc.name).write_text(doc.text, encoding="utf-8", newline="\n")
        after = documents + added
        details["added_documents"] = [doc.name for doc in added]
    elif name == "delete-5pct":
        selected = _select(seed, name, documents, 5)
        for doc in selected:
            (data / doc.name).unlink()
        removed = {doc.name for doc in selected}
        after = [doc for doc in documents if doc.name not in removed]
        details["deleted_documents"] = sorted(removed)
    elif name == "rename-5pct":
        selected = {doc.name for doc in _select(seed, name, documents, 5)}
        after = []
        for doc in documents:
            if doc.name in selected:
                renamed = Document(doc.name.replace(".txt", "-renamed.txt"), doc.topic, doc.text)
                (data / doc.name).rename(data / renamed.name)
                doc = renamed
            after.append(doc)
        details["renamed_documents"] = sorted(selected)
    elif name == "metadata-only":
        selected = _select(seed, name, documents, 10)
        for doc in selected:
            path = data / doc.name
            stat = path.stat()
            os.utime(path, (stat.st_atime + 86400, stat.st_mtime + 86400))
        details["touched_documents"] = [doc.name for doc in selected]
    elif name in CONFIG_SCENARIOS:
        current = config.load(workspace).data
        size, overlap = current["chunking"]["chunk_size"], current["chunking"]["chunk_overlap"]
        if name == "chunk-size":
            new_size = max(overlap + 1, math.floor(size * 0.8 + 0.5))
            _set(workspace, "chunking.chunk_size", new_size)
            details["config_changes"] = {"chunking.chunk_size": [size, new_size]}
        elif name == "chunk-overlap":
            new_overlap = min(size - 1, overlap * 2 if overlap else max(1, size // 10))
            _set(workspace, "chunking.chunk_overlap", new_overlap)
            details["config_changes"] = {"chunking.chunk_overlap": [overlap, new_overlap]}
        elif name == "preprocessing":
            lowercase = current["preprocessing"]["lowercase"]
            _set(workspace, "preprocessing.lowercase", not lowercase)
            details["config_changes"] = {"preprocessing.lowercase": [lowercase, not lowercase]}
        else:
            if model_change is None:
                raise BenchmarkError("model-change requires --model-change-config")
            changes = {}
            for key in MODEL_KEYS:
                if key in model_change and model_change[key] != current["embedding"].get(key):
                    changes[f"embedding.{key}"] = [current["embedding"].get(key), model_change[key]]
                    _set(workspace, f"embedding.{key}", model_change[key])
            details["config_changes"] = changes
    else:
        raise BenchmarkError(f"Unknown scenario: {name}")
    details["documents_after"] = len(after)
    return after, details
