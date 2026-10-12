"""Instrumentation shared by every benchmark strategy.

All strategies call the model through one `EncoderPool`, so a model is loaded
once (cold load is timed separately) and every input actually sent to it is
counted. Caches therefore cannot hide work from a baseline.
"""

import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import staging_config
from ..chunking import split_text
from ..document_loader import load
from ..embedding_engine import encode_batch
from ..hashing import hash_bytes, hash_payload, occurrence_id

VECTOR_TOLERANCE = 1e-4
"""Maximum absolute per-component difference accepted as numerically equal."""


class BenchmarkError(Exception):
    """The benchmark cannot run or a required step failed."""


class CountingEncoder:
    """Pass-through encoder that records calls, inputs and encode time."""

    def __init__(self, inner):
        self.inner = inner
        self.dimension = inner.dimension
        self.provenance = inner.provenance
        self.reset()

    def reset(self):
        self.calls = 0
        self.inputs = 0
        self.seconds = 0.0

    def encode(self, texts, *, batch_size, normalize_embeddings):
        started = time.perf_counter()
        try:
            return self.inner.encode(texts, batch_size=batch_size, normalize_embeddings=normalize_embeddings)
        finally:
            self.seconds += time.perf_counter() - started
            self.calls += 1
            self.inputs += len(texts)


class EncoderPool:
    """Callable used as `encoder_factory`; loads each model configuration once."""

    def __init__(self, factory):
        self.factory = factory
        self.encoders: dict[str, CountingEncoder] = {}
        self.load_seconds: dict[str, float] = {}

    @staticmethod
    def key(embedding: dict) -> str:
        return hash_payload({name: embedding.get(name) for name in ("model", "revision", "pooling", "dimension")})

    def __call__(self, embedding: dict) -> CountingEncoder:
        key = self.key(embedding)
        if key not in self.encoders:
            started = time.perf_counter()
            self.encoders[key] = CountingEncoder(self.factory(embedding))
            self.load_seconds[key] = time.perf_counter() - started
        return self.encoders[key]

    def reset(self):
        for encoder in self.encoders.values():
            encoder.reset()

    def totals(self) -> dict:
        return {"calls": sum(e.calls for e in self.encoders.values()),
                "inputs": sum(e.inputs for e in self.encoders.values()),
                "seconds": sum(e.seconds for e in self.encoders.values())}


class SyncTimer:
    """Wraps an adapter factory to time vector-store replacement separately."""

    def __init__(self, backend):
        self.backend = backend
        self.seconds = 0.0
        self.collections: list[str] = []

    def __call__(self, root, settings):
        adapter = self.backend.adapter(root, settings)
        timer = self

        class Timed:
            def replace(self, commit, records, dimension):
                started = time.perf_counter()
                try:
                    name = adapter.replace(commit, records, dimension)
                    timer.collections.append(name)
                    return name
                finally:
                    timer.seconds += time.perf_counter() - started
        return Timed()


class ChromaBackend:
    """One persistent Chroma client for every workspace in a benchmark run."""

    def __init__(self, path: Path):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise BenchmarkError('The benchmark needs Chroma. Install it with: pip install -e ".[vector-store]"') from exc
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))

    def adapter(self, root, settings):
        from ..vector_store.chromadb_adapter import ChromaAdapter
        return ChromaAdapter(root, settings, client=self.client)

    def fetch(self, name: str) -> dict[str, tuple[str, list[float]]]:
        collection = self.client.get_collection(name)
        page = collection.get(include=["embeddings", "documents"])
        result = {}
        for i, identifier in enumerate(page["ids"]):
            vector = page["embeddings"][i]
            result[identifier] = (page["documents"][i], vector.tolist() if hasattr(vector, "tolist") else list(vector))
        return result

    def reset(self):
        for collection in self.client.list_collections():
            self.client.delete_collection(getattr(collection, "name", collection))


@dataclass
class Occurrence:
    id: str
    source: str
    ordinal: int
    text: str


@dataclass
class ScannedDocument:
    name: str
    content_hash: str
    occurrences: list[Occurrence] = field(default_factory=list)


def source_files(root: Path, settings) -> dict[str, Path]:
    directory = root / settings.source_directory
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in settings.readable_extensions:
            files[path.relative_to(root).as_posix()] = path
    return files


def chunk_document(name: str, path: Path, settings, raw: bytes | None = None) -> ScannedDocument:
    raw = path.read_bytes() if raw is None else raw
    document = load(path, settings, raw=raw)
    chunks = split_text(document.text, settings.chunking.chunk_size, settings.chunking.chunk_overlap)
    return ScannedDocument(name, hash_bytes(raw), [
        Occurrence(occurrence_id(name, ordinal), name, ordinal, chunk.text) for ordinal, chunk in enumerate(chunks)])


def load_settings(root: Path):
    return staging_config.load(root)


def encode_all(encoder, texts: list[str], embedding: dict) -> list[list[float]]:
    """Encode in configured batches exactly as `embeddingvc embed` does."""
    vectors = []
    size = embedding["batch_size"]
    for offset in range(0, len(texts), size):
        vectors.extend(encode_batch(encoder, texts[offset:offset + size], batch_size=size,
                                    normalize_embeddings=embedding["normalize_embeddings"]))
    return vectors


def rows_from(occurrences: list[Occurrence], vectors: list[list[float]]) -> list[dict]:
    return [{"id": occ.id, "source": occ.source, "text": occ.text, "vector": vector,
             "metadata": {"source": occ.source, "ordinal": occ.ordinal}}
            for occ, vector in zip(occurrences, vectors)]


def compare_states(rows: list[dict], reference: list[dict], tolerance: float = VECTOR_TOLERANCE) -> dict:
    """Membership, text and vector agreement between two final states."""
    left = {row["id"]: row for row in rows}
    right = {row["id"]: row for row in reference}
    shared = left.keys() & right.keys()
    deviation = 0.0
    for identifier in shared:
        a, b = left[identifier]["vector"], right[identifier]["vector"]
        if len(a) != len(b):
            deviation = math.inf
            break
        deviation = max(deviation, max((abs(x - y) for x, y in zip(a, b)), default=0.0))
    texts_equal = all(left[i]["text"] == right[i]["text"] for i in shared)
    membership_equal = left.keys() == right.keys()
    return {"membership_equal": membership_equal, "texts_equal": texts_equal,
            "max_abs_vector_difference": deviation if math.isfinite(deviation) else None,
            "tolerance": tolerance,
            "equivalent": membership_equal and texts_equal and math.isfinite(deviation) and deviation <= tolerance}


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def process_memory() -> dict:
    """Current and lifetime-peak resident memory of this process, when available.

    The peak is monotonic over the process lifetime (operating systems do not
    offer a per-interval peak), so it bounds rather than isolates one run.
    """
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
                return {"rss_bytes": counters.WorkingSetSize, "peak_rss_bytes": counters.PeakWorkingSetSize,
                        "method": "GetProcessMemoryInfo working set"}
        else:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            scale = 1 if sys.platform == "darwin" else 1024
            return {"rss_bytes": None, "peak_rss_bytes": peak * scale, "method": "getrusage ru_maxrss"}
    except Exception:  # Memory is optional evidence; never fail a run over it.
        pass
    return {"rss_bytes": None, "peak_rss_bytes": None, "method": "unavailable"}
