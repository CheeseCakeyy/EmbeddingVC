"""Lay the commit tree out as lanes, newest first, with main on the first lane.

Commits have one parent, so branches only ever fork: lanes join but never merge back.
"""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    cells: list[tuple[str, int]]  # (text, lane), drawn left to right.
    commit: dict | None = None
    hidden: int = 0  # Older commits this row stands in for.


def ordered(graph: dict) -> list[dict]:
    """Children always come before their parents; otherwise newest first."""
    commits = {entry["commit"]: entry for entry in graph["commits"]}
    children = Counter(entry["parent"] for entry in commits.values() if entry["parent"])
    ready = [entry for entry in commits.values() if not children[entry["commit"]]]
    result = []
    while ready:
        ready.sort(key=lambda entry: entry["timestamp"], reverse=True)
        entry = ready.pop(0)
        result.append(entry)
        parent = entry["parent"]
        if parent in commits:
            children[parent] -= 1
            if not children[parent]:
                ready.append(commits[parent])
    return result


def _join(lanes: list, hits: list[int]) -> Row:
    """Draw the row where branch lanes run back into the commit they forked from."""
    first, last = hits[0], hits[-1]
    cells = []
    for lane, value in enumerate(lanes):
        if lane < first or lane > last:
            cells.append(("│ " if value else "  ", lane))
        elif lane == first:
            cells += [("├", lane), ("─", lane + 1)]
        elif lane == last:
            cells.append(("╯ ", lane))
        elif lane in hits:
            cells += [("┴", lane), ("─", lane + 1)]
        else:
            cells += [("┼" if value else "─", lane), ("─", lane + 1)]
    return Row(cells)


def _trim(lanes: list) -> None:
    while lanes and lanes[-1] is None:
        lanes.pop()


def rows(graph: dict, limit: int | None = None) -> list[Row]:
    order = ordered(graph)
    main = graph["tips"].get("main")
    lanes = [main] if main else []
    result = []
    for position, entry in enumerate(order):
        if limit is not None and position == limit:
            cells = [("┆ " if value else "  ", lane) for lane, value in enumerate(lanes)]
            result.append(Row(cells, hidden=len(order) - limit))
            break
        digest = entry["commit"]
        hits = [lane for lane, value in enumerate(lanes) if value == digest]
        if len(hits) > 1:
            result.append(_join(lanes, hits))
            for lane in hits[1:]:
                lanes[lane] = None
            _trim(lanes)
        if hits:
            column = hits[0]
        elif None in lanes:
            column = lanes.index(None)
        else:
            lanes.append(None)
            column = len(lanes) - 1
        cells = [("● " if lane == column else "│ " if value else "  ", lane) for lane, value in enumerate(lanes)]
        result.append(Row(cells, entry))
        lanes[column] = entry["parent"]
        _trim(lanes)
    return result
