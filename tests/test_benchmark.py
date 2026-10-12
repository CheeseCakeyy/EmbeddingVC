import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from embeddingvc.benchmarks import evaluate_retrieval as metrics
from embeddingvc.benchmarks import generate_dataset
from embeddingvc.benchmarks.common import BenchmarkError
from embeddingvc.benchmarks.scenarios import changed_count
from embeddingvc.cli import app
from embeddingvc.commands.benchmark import benchmark
from embeddingvc.commands.config import set_
from embeddingvc.hashing import hash_text
from embeddingvc.repository import initialize

REVISION = "a" * 40


class Encoder:
    """Deterministic bag-of-words vectors, so related texts are similar."""

    loads = 0

    def __init__(self, settings):
        Encoder.loads += 1
        self.dimension = 16
        self.provenance = {"provider": "fake", "model": settings["model"], "revision": settings["revision"]}
        self.salt = settings["model"]

    def encode(self, texts, *, batch_size, normalize_embeddings):
        vectors = []
        for text in texts:
            values = [0.0] * self.dimension
            for word in text.lower().split():
                values[int(hash_text(self.salt + word.strip(".,"))[:4], 16) % self.dimension] += 1.0
            norm = math.sqrt(sum(v * v for v in values)) or 1.0
            vectors.append([v / norm for v in values] if normalize_embeddings else values)
        return vectors


class MemoryBackend:
    def __init__(self, path):
        self.collections = {}

    def adapter(self, root, settings):
        backend = self

        class Adapter:
            def replace(self, commit, records, dimension):
                name = f"memory-{len(backend.collections)}-{commit[:12]}"
                backend.collections[name] = {row["id"]: (row["text"], list(row["vector"])) for row in records}
                return name
        return Adapter()

    def fetch(self, name):
        return self.collections[name]

    def reset(self):
        self.collections.clear()


class MetricTests(unittest.TestCase):
    def test_known_ranking(self):
        ranked, relevant = ["a", "x", "b", "y"], {"a", "b", "c"}
        self.assertEqual(metrics.precision_at_k(ranked, relevant, 2), 0.5)
        self.assertAlmostEqual(metrics.recall_at_k(ranked, relevant, 3), 2 / 3)
        self.assertEqual(metrics.reciprocal_rank(["x", "a"], relevant), 0.5)
        ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(metrics.ndcg_at_k(ranked, relevant, 3), (1 + 1 / math.log2(4)) / ideal)
        self.assertEqual(metrics.ndcg_at_k(["a", "b", "c"], relevant, 3), 1.0)

    def test_missing_and_zero_cases_are_not_reported_as_zero(self):
        self.assertIsNone(metrics.recall_at_k(["a"], set(), 5))
        self.assertIsNone(metrics.ndcg_at_k(["a"], set(), 5))
        self.assertEqual(metrics.reciprocal_rank(["a"], {"z"}), 0.0)
        self.assertIsNone(metrics.top_k_overlap([], [], 5))
        self.assertEqual(metrics.top_k_overlap(["a", "b"], ["b", "c"], 2), 0.5)

    def test_chunks_collapse_to_documents_and_zero_vectors_do_not_crash(self):
        rows = [{"id": "1", "source": "d1", "vector": [1.0, 0.0]},
                {"id": "2", "source": "d1", "vector": [0.9, 0.1]},
                {"id": "3", "source": "d2", "vector": [0.0, 1.0]},
                {"id": "4", "source": "d3", "vector": [0.0, 0.0]}]
        self.assertEqual(metrics.rank_documents([1.0, 0.0], rows), ["d1", "d2", "d3"])


class DatasetTests(unittest.TestCase):
    def test_same_seed_reproduces_inputs(self):
        self.assertEqual(generate_dataset.corpus(7, 30), generate_dataset.corpus(7, 30))
        self.assertNotEqual(generate_dataset.corpus(7, 30), generate_dataset.corpus(8, 30))
        first = generate_dataset.manifest(7, generate_dataset.corpus(7, 30))
        self.assertEqual(first, generate_dataset.manifest(7, generate_dataset.corpus(7, 30)))

    def test_every_query_has_relevant_documents_and_chunks_are_unique(self):
        corpus = generate_dataset.corpus(3, 24)
        self.assertTrue(all(generate_dataset.relevance(corpus).values()))
        paragraphs = [p for doc in corpus for p in doc.text.split("\n\n")]
        self.assertEqual(len(paragraphs), len(set(paragraphs)))

    def test_deterministic_rounding(self):
        self.assertEqual([changed_count(p, 20) for p in (1, 5, 10)], [1, 1, 2])
        self.assertEqual(changed_count(5, 30), 2)  # 1.5 rounds half up


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = initialize(Path(temp.name) / "repo").resolve()
        set_(self.root, "model_revision", REVISION)
        set_(self.root, "chunk_size", "300")
        Encoder.loads = 0

    def run_benchmark(self, **kwargs):
        options = {"seed": 5, "repeats": 1, "documents": 24, "top_k": 3, "warmup": 2,
                   "encoder_factory": Encoder, "backend_factory": MemoryBackend}
        options.update(kwargs)
        return benchmark(self.root, **options)

    def results(self, result):
        return json.loads((result.directory / "raw_results.json").read_text(encoding="utf-8"))

    def by(self, raw, scenario, strategy):
        return next(r for r in raw["runs"] if r["scenario"] == scenario and r["strategy"] == strategy)

    def test_regeneration_policies_and_reports(self):
        repository = {p: p.read_bytes() for p in (self.root / ".embeddingvc").rglob("*") if p.is_file()}
        result = self.run_benchmark(scenarios=["edit-5pct", "metadata-only", "rename-5pct", "chunk-size", "model-change"])
        self.assertEqual(result.successful, 4)
        self.assertEqual(result.failures, [])
        self.assertEqual([s["scenario"] for s in result.skipped], ["model-change"])
        for name in ("environment.json", "config.json", "dataset_manifest.json", "queries.json",
                     "raw_results.json", "summary.md", "failures.json"):
            self.assertTrue((result.directory / name).is_file(), name)
        raw = self.results(result)
        edit = {s: self.by(raw, "edit-5pct", s) for s in ("full", "document", "chunk")}
        self.assertEqual(edit["full"]["model_inputs"], edit["full"]["occurrences"])
        self.assertLess(edit["document"]["model_inputs"], edit["full"]["model_inputs"])
        self.assertLess(edit["chunk"]["model_inputs"], edit["document"]["model_inputs"])
        self.assertTrue(edit["chunk"]["equivalence_to_full"]["equivalent"])
        self.assertTrue(edit["document"]["equivalence_to_full"]["equivalent"])
        self.assertEqual(edit["chunk"]["retrieval"]["identical_top_k_fraction"], 1.0)
        self.assertEqual(self.by(raw, "metadata-only", "chunk")["model_inputs"], 0)
        self.assertEqual(self.by(raw, "metadata-only", "document")["model_inputs"], 0)
        # Renamed files have identical chunk text: content addressing reuses it.
        self.assertEqual(self.by(raw, "rename-5pct", "chunk")["model_inputs"], 0)
        self.assertGreater(self.by(raw, "rename-5pct", "document")["model_inputs"], 0)
        size = {s: self.by(raw, "chunk-size", s) for s in ("full", "document", "chunk")}
        self.assertEqual(size["document"]["model_inputs"], size["full"]["model_inputs"])
        self.assertTrue(all(item["equivalent"] and item["encoder_inputs_during_checkout"] == 0
                            for item in raw["reproducibility"]))
        self.assertTrue(all(all(r["policy_check"].values()) for r in raw["runs"] if r.get("policy_check")))
        self.assertEqual(Encoder.loads, 1)  # Loaded once; never per strategy.
        summary = (result.directory / "summary.md").read_text(encoding="utf-8")
        self.assertIn("model-change: no --model-change-config given", summary)
        self.assertNotIn("| NO", summary)
        after = {p: p.read_bytes() for p in (self.root / ".embeddingvc").rglob("*") if p.is_file()}
        self.assertEqual(after, repository)
        remaining = {path.name for path in (result.directory / "workspaces").iterdir()}
        self.assertLessEqual(remaining, {"baseline", "chroma"})  # Scenario copies are removed.

    def test_model_change_with_second_configuration(self):
        path = self.root / "second.yaml"
        path.write_text(f"embedding:\n  model: other/model\n  revision: {'b' * 40}\n", encoding="utf-8")
        result = self.run_benchmark(scenarios=["model-change"], model_change_config=str(path))
        raw = self.results(result)
        chunk = self.by(raw, "model-change", "chunk")
        self.assertEqual(result.skipped, [])
        self.assertEqual(chunk["reuse_rate"], 0.0)
        self.assertTrue(chunk["equivalence_to_full"]["equivalent"])
        self.assertEqual(Encoder.loads, 2)

    def test_same_seed_gives_same_inputs_in_unique_directories(self):
        first = self.run_benchmark(scenarios=["edit-1pct"])
        second = self.run_benchmark(scenarios=["edit-1pct"])
        self.assertNotEqual(first.directory, second.directory)
        manifests = [json.loads((r.directory / "dataset_manifest.json").read_text()) for r in (first, second)]
        self.assertEqual(manifests[0], manifests[1])
        a, b = (self.results(r)["scenarios"] for r in (first, second))
        self.assertEqual(a, b)

    def test_failures_are_recorded_and_baseline_requires_pinned_revision(self):
        set_(self.root, "model_revision", "null")
        with self.assertRaisesRegex(BenchmarkError, "40-character"):
            self.run_benchmark(scenarios=["edit-1pct"])
        with self.assertRaisesRegex(BenchmarkError, "Unknown scenario"):
            self.run_benchmark(scenarios=["nope"])
        bad = self.root / "bad.yaml"
        bad.write_text("embedding:\n  model: x\n  revision: main\n", encoding="utf-8")
        with self.assertRaisesRegex(BenchmarkError, "model-change-config"):
            self.run_benchmark(model_change_config=str(bad))

    def test_cli_rejects_bad_arguments_and_unknown_scenarios(self):
        for argv in (["benchmark", "--repeats", "0"], ["benchmark", "--scenarios", "edit-2pct"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                app(argv)
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
