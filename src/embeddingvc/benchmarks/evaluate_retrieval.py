"""Exact (brute-force) retrieval and ranking metrics over document relevance.

Ranking is exact cosine similarity, not an approximate index, so metric
differences come from the vectors themselves rather than from index noise.
Chunks are ranked first and collapsed to documents in first-hit order.
"""

import math


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(item in relevant for item in ranked[:k]) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None  # Undefined, never reported as zero.
    return sum(item in relevant for item in ranked[:k]) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1 / position
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    dcg = sum(1 / math.log2(position + 1) for position, item in enumerate(ranked[:k], start=1)
              if item in relevant)
    ideal = sum(1 / math.log2(position + 1) for position in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal


def top_k_overlap(left: list[str], right: list[str], k: int) -> float | None:
    a, b = set(left[:k]), set(right[:k])
    if not a and not b:
        return None
    return len(a & b) / max(len(a), len(b))


def _unit(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else None


def rank_documents(query: list[float], rows: list[dict]) -> list[str]:
    """Rank documents by their best chunk's cosine similarity to the query."""
    if not rows:
        return []
    try:
        import numpy
    except ModuleNotFoundError:  # pragma: no cover - numpy ships with the extras
        numpy = None
    if numpy is not None:
        matrix = numpy.asarray([row["vector"] for row in rows], dtype=numpy.float64)
        norms = numpy.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        q = numpy.asarray(query, dtype=numpy.float64)
        q_norm = numpy.linalg.norm(q) or 1.0
        scores = (matrix @ q) / (norms * q_norm)
        scores = scores.tolist()
    else:
        unit_query = _unit(query) or query
        scores = []
        for row in rows:
            unit_row = _unit(row["vector"])
            scores.append(sum(a * b for a, b in zip(unit_row, unit_query)) if unit_row else 0.0)
    # Ties break on occurrence id so rankings are deterministic.
    order = sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i]["id"]))
    ranked, seen = [], set()
    for i in order:
        source = rows[i]["source"]
        if source not in seen:
            seen.add(source)
            ranked.append(source)
    return ranked


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def evaluate(query_vectors: dict[str, list[float]], rows: list[dict], labels: dict[str, list[str]],
             k: int, reference: dict[str, list[str]] | None = None) -> dict:
    """Return mean metrics plus per-query rankings for later comparison."""
    rankings, per_query = {}, []
    for query_id in sorted(query_vectors):
        ranked = rank_documents(query_vectors[query_id], rows)
        rankings[query_id] = ranked
        relevant = set(labels.get(query_id, []))
        entry = {"query": query_id, "relevant": len(relevant),
                 "precision_at_k": precision_at_k(ranked, relevant, k),
                 "recall_at_k": recall_at_k(ranked, relevant, k),
                 "mrr": reciprocal_rank(ranked, relevant),
                 "ndcg_at_k": ndcg_at_k(ranked, relevant, k)}
        if reference is not None:
            entry["top_k_overlap"] = top_k_overlap(ranked, reference.get(query_id, []), k)
            entry["identical_top_k"] = ranked[:k] == reference.get(query_id, [])[:k]
        per_query.append(entry)
    summary = {"k": k, "queries": len(per_query),
               "queries_without_relevant": sum(entry["relevant"] == 0 for entry in per_query)}
    for key in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "top_k_overlap"):
        summary[key] = _mean(entry.get(key) for entry in per_query)
    if reference is not None:
        summary["identical_top_k_fraction"] = _mean(float(entry["identical_top_k"]) for entry in per_query)
    return {"summary": summary, "rankings": rankings, "per_query": per_query}
