"""Benchmark orchestration: isolated workspaces, measurement and reports.

The user's repository is only read (for its configuration) and receives a new
`reports/benchmarks/<run-id>/` directory. Every measured update runs in a copy
of one baseline workspace, so HEAD, index and database are never touched.
"""

import hashlib
import json
import os
import platform
import secrets
import shutil
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

from .. import __version__, commit_manager as manager, config, objects
from ..commands.add import add
from ..commands.checkout import checkout
from ..commands.commit import commit
from ..commands.embed import embed
from ..embedding_engine import EmbeddingError, validate_revision
from ..repository import initialize
from . import evaluate_retrieval, generate_dataset, scenarios
from . import run_document_regeneration, run_full_regeneration, run_incremental_regeneration
from .common import (BenchmarkError, ChromaBackend, EncoderPool, SyncTimer, VECTOR_TOLERANCE,
                     compare_states, directory_bytes, load_settings, process_memory)

STRATEGIES = ("full", "document", "chunk")
PACKAGES = ("embeddingvc", "sentence-transformers", "transformers", "torch", "tokenizers",
            "chromadb", "numpy", "PyMuPDF", "PyYAML")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def environment(pool: EncoderPool, warmup: int) -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    installed = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib_metadata.distributions()
                       if dist.metadata["Name"])
    return {"python": sys.version, "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "packages": versions,
            "installed_distributions_sha256": hashlib.sha256("\n".join(installed).encode()).hexdigest(),
            "installed_distributions": installed, "lockfile": None,
            "model_load_seconds": pool.load_seconds,
            "encoders": {key: encoder.provenance for key, encoder in pool.encoders.items()},
            "warmup_inputs_per_model": warmup,
            "timing_method": "time.perf_counter wall clock; time.process_time CPU (all threads)",
            "memory_method": process_memory()["method"]}


def _baseline_config(user_root: Path) -> dict:
    data = config.load(user_root).data
    try:
        validate_revision(data["embedding"])
    except EmbeddingError as exc:
        raise BenchmarkError(str(exc)) from exc
    data = json.loads(json.dumps(data))
    data["repository"]["name"] = "embeddingvc-benchmark"
    data["documents"]["source_directory"] = "./data"
    data["documents"]["supported_types"] = ["txt"]
    data["vector_store"]["persist_directory"] = "./output/chroma"
    data["vector_store"]["collection"] = "benchmark"
    return data


def _warm(pool: EncoderPool, embedding: dict, count: int) -> None:
    encoder = pool(embedding)
    if count:
        encoder.encode([f"warm up sentence {i}" for i in range(count)], batch_size=embedding["batch_size"],
                       normalize_embeddings=embedding["normalize_embeddings"])
    encoder.reset()


def _settle(workspace: Path) -> int:
    """Read every file of a freshly copied workspace once, outside any timing.

    The first open of a just-written file can be far slower than normal (on
    Windows, real-time antivirus scans it), which would charge copy artifacts to
    whichever strategy touches the most files. A real repository's files are
    not brand new, so all strategies are measured on settled, cached files.
    """
    total = 0
    for path in workspace.rglob("*"):
        if path.is_file():
            total += len(path.read_bytes())
    return total


def _query_vectors(pool: EncoderPool, embedding: dict, cache: dict) -> dict:
    key = EncoderPool.key(embedding)
    if key not in cache:
        queries = generate_dataset.queries()
        encoder = pool(embedding)
        vectors = encoder.encode([q["text"] for q in queries], batch_size=embedding["batch_size"],
                                 normalize_embeddings=embedding["normalize_embeddings"])
        vectors = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        cache[key] = {q["id"]: list(v) for q, v in zip(queries, vectors)}
        encoder.reset()
    return cache[key]


def _storage(workspace: Path, head: str) -> dict:
    """Versioned history on disk versus what one snapshot alone needs."""
    meta = workspace / ".embeddingvc"
    manifest = manager.read_commit(workspace, head)
    referenced = {("configs", manifest[key]) for key in ("config", "config_object", "embedding_config")}
    for doc in manifest["documents"].values():
        for occ in doc["occurrences"]:
            referenced |= {("chunks", occ["chunk"]), ("embeddings", occ["embedding"]["vector"])}
    single = sum(objects.object_path(workspace, kind, digest).stat().st_size for kind, digest in referenced)
    single += (meta / "commits" / f"{head}.json").stat().st_size
    versioned = directory_bytes(meta / "objects") + directory_bytes(meta / "commits")
    vectors = len({digest for kind, digest in referenced if kind == "embeddings"})
    return {"versioned_bytes": versioned, "single_version_bytes": single,
            "storage_overhead_bytes": versioned - single, "unique_vectors": vectors,
            "method": "file sizes of unique objects and manifests; vector database excluded"}


def _ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None


def _stats(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {"mean": statistics.fmean(values), "stdev": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values), "n": len(values)}


class Benchmark:
    def __init__(self, user_root: Path, *, seed: int, repeats: int, documents: int, top_k: int,
                 selected: list[str], model_change: dict | None, keep_workspaces: bool,
                 warmup: int, encoder_factory, backend_factory, progress):
        self.user_root, self.seed, self.repeats = user_root, seed, repeats
        self.documents, self.top_k, self.selected = documents, top_k, selected
        self.model_change, self.keep, self.warmup = model_change, keep_workspaces, warmup
        self.progress = progress or (lambda message: None)
        self.pool = EncoderPool(encoder_factory)
        self.run_id = f"{_now()}-{secrets.token_hex(3)}"
        self.directory = user_root / "reports" / "benchmarks" / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous run.
        self.workspaces = self.directory / "workspaces"
        self.workspaces.mkdir()
        self.backend = backend_factory(self.workspaces / "chroma")
        self.query_cache: dict = {}
        self.runs, self.failures, self.skipped, self.details, self.reproducibility = [], [], [], {}, []

    def build_baseline(self, corpus) -> dict:
        root = initialize(self.workspaces / "baseline")
        configuration = _baseline_config(self.user_root)
        (root / "embeddingvc.yaml").write_text(config.dump_yaml(configuration), encoding="utf-8", newline="\n")
        for doc in corpus:
            (root / "data" / doc.name).write_text(doc.text, encoding="utf-8", newline="\n")
        settings = load_settings(root)
        started = time.perf_counter()
        _warm(self.pool, settings.embedding, self.warmup)
        add([str(root / "data")], directory=root)
        embed(root, encoder_factory=self.pool)
        timer = SyncTimer(self.backend)
        head = commit(root, message="benchmark baseline", adapter_factory=timer).commit
        seconds = time.perf_counter() - started
        records, _ = manager.records(root, manager.read_commit(root, head))
        rows = [{"id": r["id"], "source": r["metadata"]["source"], "text": r["text"], "vector": r["vector"],
                 "metadata": r["metadata"]} for r in records]
        by_source: dict[str, list] = {}
        for row in rows:
            by_source.setdefault(row["source"], []).append(
                {**row, "metadata": {"source": row["source"], "ordinal": row["metadata"]["ordinal"]}})
        manifest = manager.read_commit(root, head)
        labels = generate_dataset.relevance(corpus)
        retrieval = evaluate_retrieval.evaluate(_query_vectors(self.pool, settings.embedding, self.query_cache),
                                                rows, labels, self.top_k)
        self.pool.reset()
        self.backend.reset()
        return {"root": root, "commit": head, "rows": rows, "rows_by_source": by_source,
                "content_hashes": {name: doc["content_hash"] for name, doc in manifest["documents"].items()},
                "effective_hash": config.load(root).effective_hash, "build_seconds": seconds,
                "occurrences": len(rows), "retrieval": retrieval["summary"]}

    def run_strategy(self, strategy, workspace, label, baseline, config_changed):
        self.pool.reset()
        timer = SyncTimer(self.backend)
        if strategy == "full":
            outcome = run_full_regeneration.run(workspace, self.pool, timer, label)
        elif strategy == "document":
            outcome = run_document_regeneration.run(workspace, self.pool, timer, label, baseline, config_changed)
        else:
            outcome = run_incremental_regeneration.run(workspace, self.pool, timer, label)
        model = self.pool.totals()
        outcome.update({"model_calls": model["calls"], "model_inputs": model["inputs"],
                        "embedding_seconds": model["seconds"], "sync_seconds": timer.seconds,
                        "memory": process_memory()})
        return outcome

    def run_scenario(self, name, corpus, baseline, repeat, index, total):
        label = f"{name}-r{repeat}"
        self.progress(f"[{index}/{total}] {name} (repeat {repeat + 1}/{self.repeats})")
        workspace = self.workspaces / label
        shutil.copytree(baseline["root"], workspace, ignore=shutil.ignore_patterns("output"))
        after, details = scenarios.apply(name, workspace, corpus, self.seed, self.model_change)
        self.details.setdefault(name, details)
        _settle(workspace)
        settings = load_settings(workspace)
        config_changed = config.load(workspace).effective_hash != baseline["effective_hash"]
        _warm(self.pool, settings.embedding, self.warmup)
        labels = generate_dataset.relevance(after)
        queries = _query_vectors(self.pool, settings.embedding, self.query_cache)
        order = STRATEGIES[repeat % 3:] + STRATEGIES[:repeat % 3]  # Rotate to spread order effects.
        outcomes = {}
        for position, strategy in enumerate(order):
            outcomes[strategy] = self.run_strategy(strategy, workspace, f"{label}-{strategy}", baseline, config_changed)
            outcomes[strategy]["order"] = position
        reference = evaluate_retrieval.evaluate(queries, outcomes["full"]["rows"], labels, self.top_k)
        for strategy in STRATEGIES:
            outcome = outcomes[strategy]
            evaluation = (reference if strategy == "full" else
                          evaluate_retrieval.evaluate(queries, outcome["rows"], labels, self.top_k, reference["rankings"]))
            record = {key: value for key, value in outcome.items() if key not in ("rows",)}
            record.update({"scenario": name, "repeat": repeat, "strategy": strategy, "status": "ok",
                           "retrieval": evaluation["summary"],
                           "regenerated_inputs_share": _ratio(outcome["model_inputs"], outcome["occurrences"]),
                           "regeneration_reduction": (1 - outcome["model_inputs"] / outcome["occurrences"]
                                                      if outcome["occurrences"] else None),
                           "reuse_rate": _ratio(outcome["reused_occurrences"], outcome["occurrences"]),
                           "equivalence_to_full": (None if strategy == "full" else
                                                   compare_states(outcome["rows"], outcomes["full"]["rows"]))})
            if strategy == "chunk":
                record["storage"] = _storage(workspace, outcome["commit"])
            self.runs.append(record)
        inputs = {strategy: outcomes[strategy]["model_inputs"] for strategy in STRATEGIES}
        self.runs[-1]["policy_check"] = {
            "chunk_inputs_le_document_inputs": inputs["chunk"] <= inputs["document"],
            "document_inputs_le_full_inputs": inputs["document"] <= inputs["full"],
            "full_inputs_equal_occurrences": inputs["full"] == outcomes["full"]["occurrences"]}
        if repeat == 0:
            self.check_reproducibility(name, workspace, baseline)
        self.backend.reset()
        if not self.keep:
            shutil.rmtree(workspace, ignore_errors=True)

    def check_reproducibility(self, name, workspace, baseline):
        """Restore the baseline snapshot with checkout and compare its database."""
        self.pool.reset()
        timer = SyncTimer(self.backend)
        started = time.perf_counter()
        result = checkout(baseline["commit"], workspace, adapter_factory=timer)
        seconds = time.perf_counter() - started
        stored = self.backend.fetch(timer.collections[-1])
        restored = [{"id": key, "text": text, "vector": vector} for key, (text, vector) in stored.items()]
        comparison = compare_states(restored, baseline["rows"])
        comparison.update({"scenario": name, "checkout_seconds": seconds, "records": result.records,
                           "encoder_inputs_during_checkout": self.pool.totals()["inputs"]})
        self.reproducibility.append(comparison)

    def execute(self) -> dict:
        corpus = generate_dataset.corpus(self.seed, self.documents)
        dataset = generate_dataset.manifest(self.seed, corpus)
        _write(self.directory / "dataset_manifest.json", dataset)
        _write(self.directory / "queries.json", {"queries": generate_dataset.queries(),
                                                 "baseline_relevance": generate_dataset.relevance(corpus)})
        self.progress(f"Building baseline: {len(corpus)} documents")
        baseline = self.build_baseline(corpus)
        names = [name for name in self.selected
                 if not (name == "model-change" and self.model_change is None)]
        if "model-change" in self.selected and self.model_change is None:
            self.skipped.append({"scenario": "model-change",
                                 "reason": "no --model-change-config given; a second pinned model is required"})
        total, index = len(names) * self.repeats, 0
        for repeat in range(self.repeats):
            for name in names:
                index += 1
                try:
                    self.run_scenario(name, corpus, baseline, repeat, index, total)
                except Exception as exc:
                    self.failures.append({"scenario": name, "repeat": repeat, "error": f"{type(exc).__name__}: {exc}",
                                          "traceback": traceback.format_exc()})
                    self.backend.reset()
        results = {"run_id": self.run_id, "baseline": {key: baseline[key] for key in
                   ("commit", "build_seconds", "occurrences", "retrieval")},
                   "scenarios": self.details, "runs": self.runs, "reproducibility": self.reproducibility}
        _write(self.directory / "config.json", {
            "seed": self.seed, "repeats": self.repeats, "documents": self.documents, "top_k": self.top_k,
            "scenarios": self.selected, "skipped": self.skipped, "warmup_inputs": self.warmup,
            "vector_tolerance": VECTOR_TOLERANCE, "embeddingvc_version": __version__,
            "pipeline_configuration": _baseline_config(self.user_root), "model_change": self.model_change,
            "strategy_order": "rotated by repeat: full/document/chunk, document/chunk/full, chunk/full/document"})
        _write(self.directory / "environment.json", environment(self.pool, self.warmup))
        _write(self.directory / "raw_results.json", results)
        _write(self.directory / "failures.json", {"failures": self.failures, "skipped": self.skipped})
        (self.directory / "summary.md").write_text(summary(results, self), encoding="utf-8")
        if not self.keep:
            for path in self.workspaces.iterdir():
                if path.name not in ("baseline", "chroma"):
                    shutil.rmtree(path, ignore_errors=True)
        return results


def _fmt(value, digits=3, percent=False):
    if value is None:
        return "n/a"
    if percent:
        return f"{value * 100:.1f}%"
    return f"{value:.{digits}f}"


def _mean_of(group, key):
    stats = _stats([r["retrieval"].get(key) for r in group])
    return stats["mean"] if stats else None


def _mean_sd(stats, digits=3):
    if not stats:
        return "n/a"
    sd = f" ± {stats['stdev']:.{digits}f}" if stats["stdev"] is not None else ""
    return f"{stats['mean']:.{digits}f}{sd}"


def summary(results: dict, bench: Benchmark) -> str:
    k = bench.top_k
    baseline = results["baseline"]
    lines = [f"# EmbeddingVC benchmark {results['run_id']}", "",
             f"- Seed {bench.seed}, {bench.documents} synthetic documents, {bench.repeats} repeat(s), top-k = {k}",
             f"- Baseline: {baseline['occurrences']} chunk occurrences, built in {baseline['build_seconds']:.2f}s "
             f"(commit {baseline['commit'][:12]})",
             f"- Baseline retrieval: Recall@{k} {_fmt(baseline['retrieval']['recall_at_k'])}, "
             f"MRR {_fmt(baseline['retrieval']['mrr'])}, NDCG@{k} {_fmt(baseline['retrieval']['ndcg_at_k'])}",
             "- All values below are measured in this run; see raw_results.json for every observation.", "",
             "## Coverage", "",
             f"- Completed scenario runs: {len(results['runs']) // 3}",
             f"- Failed scenario runs: {len(bench.failures)}" + (" (see failures.json)" if bench.failures else ""),
             f"- Skipped scenarios: {len(bench.skipped)}"]
    lines += [f"  - {item['scenario']}: {item['reason']}" for item in bench.skipped]
    lines += ["", "## Results (mean ± standard deviation across repeats)", "",
              "Model inputs = texts actually sent to the encoder. Reduction = 1 - inputs / occurrences. "
              "Speedup = full-regeneration total time / strategy total time. "
              "Overlap = mean top-k document overlap with full regeneration.", "",
              f"| Scenario | Strategy | Total s | Embedding s | Sync s | Model inputs | Occurrences | Reduction | "
              f"Speedup | Recall@{k} | NDCG@{k} | Overlap | Equivalent |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in [n for n in bench.selected if n in results["scenarios"]]:
        runs = {s: [r for r in results["runs"] if r["scenario"] == name and r["strategy"] == s] for s in STRATEGIES}
        full_total = _stats([r["seconds"] for r in runs["full"]])
        for strategy in STRATEGIES:
            group = runs[strategy]
            if not group:
                continue
            total = _stats([r["seconds"] for r in group])
            inputs = _stats([r["model_inputs"] for r in group])
            occurrences = _stats([r["occurrences"] for r in group])
            reduction = _stats([r["regeneration_reduction"] for r in group])
            speedup = _ratio(full_total["mean"] if full_total else None, total["mean"] if total else None)
            equivalence = [r["equivalence_to_full"]["equivalent"] for r in group if r["equivalence_to_full"]]
            lines.append("| " + " | ".join([
                name, strategy, _mean_sd(total), _mean_sd(_stats([r["embedding_seconds"] for r in group])),
                _mean_sd(_stats([r["sync_seconds"] for r in group])), _mean_sd(inputs, 0), _mean_sd(occurrences, 0),
                _fmt(reduction["mean"] if reduction else None, percent=True), _fmt(speedup, 2),
                _fmt(_mean_of(group, "recall_at_k")), _fmt(_mean_of(group, "ndcg_at_k")),
                "reference" if strategy == "full" else _fmt(_mean_of(group, "top_k_overlap")),
                "reference" if strategy == "full" else ("yes" if equivalence and all(equivalence) else "NO")]) + " |")
    storage = [r["storage"] for r in results["runs"] if r.get("storage")]
    if storage:
        lines += ["", "## Versioned storage (chunk-level workspaces)", "",
                  "| Scenario | Versioned bytes | Single-version bytes | Overhead bytes |", "|---|---|---|---|"]
        seen = set()
        for record in results["runs"]:
            if record.get("storage") and record["scenario"] not in seen:
                seen.add(record["scenario"])
                s = record["storage"]
                lines.append(f"| {record['scenario']} | {s['versioned_bytes']} | {s['single_version_bytes']} | "
                             f"{s['storage_overhead_bytes']} |")
    if results["reproducibility"]:
        lines += ["", "## Checkout reproducibility (restore baseline after each scenario)", "",
                  "| Scenario | Records | Equivalent | Max abs vector diff | Encoder inputs | Checkout s |",
                  "|---|---|---|---|---|---|"]
        for item in results["reproducibility"]:
            lines.append(f"| {item['scenario']} | {item['records']} | {'yes' if item['equivalent'] else 'NO'} | "
                         f"{_fmt(item['max_abs_vector_difference'], 2)} | {item['encoder_inputs_during_checkout']} | "
                         f"{item['checkout_seconds']:.2f} |")
    checks = [r["policy_check"] for r in results["runs"] if r.get("policy_check")]
    violations = sum(not all(check.values()) for check in checks)
    lines += ["", "## Method notes", "",
              f"- Regeneration-policy counter checks passed in {len(checks) - violations}/{len(checks)} scenario runs "
              "(chunk inputs <= document inputs <= full inputs = occurrences).",
              "- Chunk-level timing is the real `embed` + `commit` commands, including rescanning, hashing, "
              "object verification and commit publication; baselines are minimal pipelines without history.",
              "- Every strategy synchronizes a fresh Chroma collection; sync time is reported separately.",
              "- Retrieval uses exact cosine ranking over committed vectors, collapsing chunks to documents.",
              f"- Vectors are compared within an absolute tolerance of {VECTOR_TOLERANCE}.",
              "- Model loading and warm-up are excluded from update timings (see environment.json).",
              "- Each copied workspace is read once before timing, so first-open costs of new files "
              "(e.g. antivirus scanning on Windows) are not charged to any strategy.",
              "- Memory is the process working set; its peak is monotonic over the whole run.",
              "- Paid-API cost is not applicable: the model runs locally.",
              "- The corpus and queries are synthetic; results may not transfer to real collections."]
    return "\n".join(lines) + "\n"
