# Benchmark

`embeddingvc benchmark` measures how much work chunk-level reuse saves compared
with two baselines, and checks that it does not change retrieval results.

```text
embeddingvc benchmark [--seed 42] [--repeats 3] [--documents 200] [--top-k 10]
                      [--scenarios edit-5pct,chunk-size] [--warmup 8]
                      [--model-change-config second-model.yaml] [--keep-workspaces]
```

Run it inside an initialized repository whose `embedding.revision` is pinned.
The repository's own documents, HEAD, index and database are never used or
changed; only its pipeline configuration (model, chunking, preprocessing) is
copied into the benchmark workspaces.

## What a run does

1. Generates a seeded synthetic corpus: documents on 12 topics plus 36 labelled
   queries. A query is relevant to every document of its topic.
2. Builds a baseline workspace (`init`, `add`, `embed`, `commit`).
3. For each scenario and repeat, copies the baseline, applies the change and
   updates it three ways:

   | Strategy | What it re-embeds |
   |---|---|
   | `full` | Every chunk occurrence, reusing nothing |
   | `document` | Every chunk of each new or changed document (all documents after a configuration change) |
   | `chunk` | The real `embeddingvc embed` + `commit`: only added, modified or stale chunks |

   Strategy order rotates between repeats. Model loading and warm-up are
   excluded from update timings. Each copied workspace is read once before
   timing: on Windows, the first open of a just-written file can be slowed by
   real-time antivirus scanning, which would otherwise be charged to the
   strategy that opens the most files.
4. Measures total, embedding and vector-store sync time, CPU time, model calls
   and inputs, reused occurrences, process memory, and versioned storage.
5. Evaluates the same queries against each final state with exact cosine
   ranking: Precision@K, Recall@K, MRR, NDCG@K, top-k overlap with full
   regeneration, and whether vectors and membership match it within 1e-4.
6. After the first repeat of each scenario, checks out the baseline commit and
   verifies the restored Chroma collection matches it without encoder calls.

## Scenarios

| Scenario | Change |
|---|---|
| `edit-1pct`, `edit-5pct`, `edit-10pct` | Rewrite one paragraph in that share of documents |
| `add-5pct`, `delete-5pct` | Add or delete documents |
| `rename-5pct` | Rename files without changing their content |
| `metadata-only` | Touch file modification times only |
| `chunk-size`, `chunk-overlap` | Chunk size x0.8, or overlap doubled |
| `preprocessing` | Toggle lowercasing |
| `model-change` | Switch to the model in `--model-change-config` |

Selection counts round half up with a minimum of one document. Without
`--model-change-config` the model-change scenario is reported as skipped, never
as zero. The file needs a second pinned model:

```yaml
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  revision: <40-character commit SHA>
```

## Output

`reports/benchmarks/<UTC time>-<suffix>/` (a run never overwrites another):

- `summary.md` - result tables, coverage, storage and reproducibility checks
- `raw_results.json` - every measured run; every summary value derives from it
- `environment.json` - Python, OS, CPU, package versions, model provenance,
  installed-distribution checksum, model load times
- `config.json`, `dataset_manifest.json`, `queries.json`, `failures.json`
- `workspaces/baseline/` - the baseline repository (scenario copies are deleted
  unless `--keep-workspaces` is given)

The command exits with status 1 if any scenario run failed; failures keep
their tracebacks in `failures.json`.

## Reading the results

- Reduction = 1 - model inputs / chunk occurrences.
- Speedup = full-regeneration total time / strategy total time.
- Chunk-level time includes EmbeddingVC's own overhead (rescanning, hashing,
  verifying stored objects and publishing a commit), so its speedup is usually
  smaller than its reduction in model calls, especially for small corpora.
- Memory is the process working set; its peak covers the whole process, so it
  is an upper bound rather than a per-strategy figure.
- The corpus is synthetic. Report results as measured on this corpus and
  hardware.
