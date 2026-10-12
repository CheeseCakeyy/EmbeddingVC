"""`embeddingvc benchmark` - compare full, document-level and chunk-level regeneration."""

from dataclasses import dataclass
from pathlib import Path

from .. import objects
from ..benchmarks.common import BenchmarkError, ChromaBackend
from ..benchmarks.runner import Benchmark
from ..benchmarks.scenarios import SCENARIOS, load_model_change
from ..config import ConfigurationError


@dataclass(frozen=True)
class Result:
    directory: Path
    root: Path
    successful: int
    failures: list
    skipped: list

    def render(self) -> str:
        lines = [f"Benchmark complete: {self.successful} successful scenario runs, {len(self.failures)} failed"]
        lines.append(f"Skipped scenarios: {len(self.skipped)}" + "".join(
            f"\n  - {item['scenario']}: {item['reason']}" for item in self.skipped))
        for failure in self.failures:
            lines.append(f"Failed: {failure['scenario']} (repeat {failure['repeat'] + 1}): {failure['error']}")
        lines.append(f"Results: {self.directory.relative_to(self.root).as_posix()}/")
        lines.append("See summary.md for measured results and evaluation coverage.")
        return "\n".join(lines)


def benchmark(directory: Path | str = ".", *, seed: int = 42, repeats: int = 3, documents: int = 200,
              top_k: int = 10, scenarios: list[str] | None = None, model_change_config: str | None = None,
              keep_workspaces: bool = False, warmup: int = 8, encoder_factory=None, backend_factory=None,
              progress=None) -> Result:
    """Run isolated experiments; the repository's HEAD, index and database are untouched.

    encoder_factory and backend_factory are test seams; the CLI uses the
    pinned Sentence Transformers model and a dedicated Chroma directory.
    """
    for name, value in (("repeats", repeats), ("documents", documents), ("top_k", top_k)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise BenchmarkError(f"{name} must be a positive integer")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise BenchmarkError("warmup must be zero or a positive integer")
    selected = list(scenarios or SCENARIOS)
    unknown = [name for name in selected if name not in SCENARIOS]
    if unknown:
        raise BenchmarkError(f"Unknown scenario(s): {', '.join(unknown)}. Choose from: {', '.join(SCENARIOS)}")
    selected = [name for name in SCENARIOS if name in selected]  # Canonical order, no duplicates.
    model_change = load_model_change(model_change_config)
    try:
        root = objects.find_repository(Path(directory))
        if encoder_factory is None:
            from ..embedding_engine import SentenceTransformerEncoder
            encoder_factory = SentenceTransformerEncoder
        run = Benchmark(root, seed=seed, repeats=repeats, documents=documents, top_k=top_k, selected=selected,
                        model_change=model_change, keep_workspaces=keep_workspaces, warmup=warmup,
                        encoder_factory=encoder_factory, backend_factory=backend_factory or ChromaBackend,
                        progress=progress)
        run.execute()
    except (objects.RepositoryError, ConfigurationError) as exc:
        raise BenchmarkError(str(exc)) from exc
    except BenchmarkError:
        raise
    except Exception as exc:  # The baseline must succeed before any scenario can be measured.
        raise BenchmarkError(f"Benchmark baseline failed: {type(exc).__name__}: {exc}") from exc
    return Result(run.directory, root, len(run.runs) // 3, run.failures, run.skipped)
