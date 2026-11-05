"""Turn command results into Rich renderables; these functions never touch a repository."""
import re
from statistics import fmean

from rich import box
from rich.align import Align
from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from . import mascot, tree
from .. import __version__
from ..benchmarks.runner import STRATEGIES

# Chalk pastels carry the information; orange marks what matters. Written for a dark terminal.
ORANGE, GRAY, WHITE = "#d97757", "#9c978c", "#efeadf"
MINT, SKY, BUTTER, ROSE, LILAC = "#a8d8b9", "#a7c7e7", "#f3dfa2", "#eba6a9", "#c9b6e4"
THEME = Theme({"accent": f"bold {ORANGE}", "orange": ORANGE, "muted": GRAY, "chalk": WHITE, "good": MINT, "warn": BUTTER,
               "bad": ROSE, "added": MINT, "modified": BUTTER, "deleted": ROSE, "stale": LILAC, "unchanged": GRAY,
               "metadata_changed": SKY, "sha": BUTTER})
LANES = (SKY, MINT, LILAC, ROSE, BUTTER)
LEFT_COLUMN = 35
SINGLE_COLUMN_BELOW = 84
WELCOME_TREE_ROWS = 4
TIP_WIDTH = 12
TREE_ROWS = 30
BAR_WIDTH = 28
DRIFT_ROWS = 8
STRATEGY_LABELS = {"full": "full", "document": "document", "chunk": "chunk (EmbeddingVC)"}


def shell_text(text: str) -> str:
    """Point hints written for the plain CLI at the matching slash command."""
    return re.sub(r"\bembeddingvc (?=[a-z])", "/", text)


def bytes_label(count: int | float) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def share_bar(part: float, whole: float, *, style: str = "good", width: int = BAR_WIDTH) -> Text:
    filled = round(width * part / whole) if whole else 0
    if part and not filled:
        filled = 1  # A nonzero value must never draw as an empty bar.
    return Text.assemble(("█" * filled, style), ("░" * (width - filled), "muted"))


def emphasize(text: Text) -> Text:
    """Pick out slash commands and commit ids wherever they appear in prose."""
    text.highlight_regex(r"(?<![\w/\\.])/[a-z]+\b", "accent")
    text.highlight_regex(r"\b(?=[a-f]*\d)[0-9a-f]{7,12}\b", "sha")
    return text


def _title(text: str) -> Text:
    return Text(text, style="accent")


def _panel(body, title: str, *, style: str = "orange") -> Panel:
    return Panel(body, title=f"[bold]{title}[/bold]", title_align="left", border_style=style, padding=(1, 2))


def tree_lines(graph: dict | None, limit: int | None = None, *, compact: bool = False) -> list[Text]:
    """One line per commit, drawn so branches visibly fork; compact keeps only what a glance needs."""
    if not graph or not graph["commits"]:
        return [Text("No commits yet", style="muted")]
    labels = {}
    for name, digest in graph["tips"].items():
        labels.setdefault(digest, []).append(name)
    lines = []
    for row in tree.rows(graph, limit):
        line = Text(no_wrap=True, overflow="ellipsis")
        entry = row.commit
        current = bool(entry) and entry["commit"] == graph["head"]
        for cell, lane in row.cells:
            node = current and cell.startswith("●")
            line.append(cell, "accent" if node else LANES[lane % len(LANES)])
        if not entry:
            line.append(f"{row.hidden} older commit{'s' if row.hidden != 1 else ''}" if row.hidden else "", "muted")
            lines.append(line)
            continue
        if not compact:
            line.append(entry["short"] + " ", "sha")
        for name in labels.get(entry["commit"], []):
            line.append(f"{name} ", "accent" if current and name == graph["branch"] else "bold")
        if current:
            where = "here" if compact else "you are here"
            line.append(f"◀ {where} " if graph["branch"] else f"◀ {where} (detached) ", "orange")
        line.append(f" {entry['message']}", "muted")
        lines.append(line)
    return lines


def tree_panel(graph: dict) -> Panel:
    return _panel(Group(*tree_lines(graph, TREE_ROWS)), "Tree")


def _tips(repository: dict | None) -> list[tuple[str, str]]:
    if not repository:
        return [("/init", "create a repository"), ("/help", "see every command")]
    if not repository["commit"]:
        return [("/add data", "track your documents"), ("/embed", "create embeddings")]
    return [("/status", "see what changed"), ("/tree", "see every branch")]


def _shortened(path: str, width: int) -> str:
    return path if len(path) <= width else "…" + path[-(width - 1):]


def welcome(repository: dict | None, graph: dict | None, *, path: str, width: int, frame: int = 0) -> Panel:
    if repository:
        head = repository["branch"] or "detached"
        summary = Text(f"{repository['model'] or 'no model configured'} · {head}", style="muted", no_wrap=True, overflow="ellipsis")
    else:
        summary = Text("No repository in this folder", style="warn")
    greeting = Text("Welcome back!", style="bold")
    location = Text(_shortened(path, LEFT_COLUMN - 2), style="muted", no_wrap=True)
    left = Group(*[Align.center(part) for part in (greeting, Text(" "), mascot.draw(frame), Text(" "), summary, location)])
    right = Group(_title("Tips for getting started"),
                  *[Text.assemble((command.ljust(TIP_WIDTH), "accent"), (note, "muted")) for command, note in _tips(repository)],
                  Text(" "), _title("Recent activity"), *tree_lines(graph, WELCOME_TREE_ROWS, compact=True))
    if width < SINGLE_COLUMN_BELOW:
        body = Group(left, Text(""), right)
    else:
        body = Table(box=box.MINIMAL, show_header=False, show_edge=False, pad_edge=False, expand=True, border_style="orange")
        body.add_column(width=LEFT_COLUMN)
        body.add_column(ratio=1, vertical="middle")
        body.add_row(left, right)
    return Panel(body, title=f"[accent]EmbeddingVC[/accent] [orange]v{__version__}[/orange]", title_align="left",
                 border_style="orange", padding=(0, 1))


def working(frame: int, progress) -> Padding:
    """The octopus beside the live progress; redrawn by the shell's Progress."""
    octopus = mascot.draw(frame)
    grid = Table.grid(padding=(0, 3))
    grid.add_row(octopus, Padding(progress, (len(octopus.plain.splitlines()) // 2, 0, 0, 0)))
    return Padding(grid, (1, 0, 0, 2))


def echo(line: str) -> Text:
    return Text.assemble(("> ", "accent"), (line, "chalk"))


def toolbar(repository: dict | None) -> str:
    if not repository:
        return " no repository · /init · /help · Ctrl+D to exit"
    head = repository["commit"][:7] if repository["commit"] else "no commits"
    return (f" {repository['branch'] or 'detached'} · {head} · {repository['documents']} docs · "
            f"{repository['chunks']} chunks · /help · Ctrl+D to exit")


def help_table(commands) -> Group:
    sections = []
    for group in dict.fromkeys(command.group for command in commands):
        table = Table.grid(padding=(0, 3))
        table.add_column(no_wrap=True, style="accent", min_width=34)
        table.add_column(overflow="fold")
        for command in commands:
            if command.group == group:
                table.add_row(Text(command.usage), Text(command.summary))
        sections += [Text(group, style="bold"), Padding(table, (0, 0, 1, 2))]
    return Group(*sections)


def message(text: str, *, style: str = "good", title: str = "Done") -> Panel:
    lines = shell_text(text).splitlines() or [""]
    body = Text("\n").join([Text(lines[0], style="bold"), *[Text(line, style="muted") for line in lines[1:]]])
    return _panel(emphasize(body), title, style=style)


def failure(text: str) -> Panel:
    return Panel(emphasize(Text(shell_text(text))), title="[bold]Error[/bold]", title_align="left", border_style="bad",
                 padding=(0, 2))


def _counts(rows: dict[str, dict[str, int]]) -> Table:
    kinds = list(dict.fromkeys(kind for counts in rows.values() for kind in counts))
    table = Table(box=None, pad_edge=False, header_style="muted")
    table.add_column("")
    for kind in kinds:
        table.add_column(kind.replace("_", " "), justify="right")
    for title, counts in rows.items():
        cells = [Text(str(counts.get(kind, 0)), style=kind if counts.get(kind) else "muted") for kind in kinds]
        table.add_row(Text(title, style="bold"), *cells)
    return table


def _next_step(report: dict) -> str | None:
    if report["checkout_incomplete"]:
        return "/checkout HEAD"
    if report["untracked"]:
        return "/add <path>"
    if report["pending"] or report["index_differs"]:
        return "/embed"
    if any(report["documents"][kind] for kind in ("added", "modified", "deleted")):
        return "/commit <message>"
    return None


def status(report: dict) -> Panel:
    required = report["pending"]
    parts = [_title(report["label"]), Text(""),
             _counts({"Documents": report["documents"], "Chunks": report["chunks"]}), Text(""),
             Text.assemble(("Embeddings required  ", "bold"),
                           (f"{required} occurrences ({report['pending_unique']} unique model inputs)",
                            "warn" if required else "good"))]
    if report["sync_status"]:
        parts.append(Text.assemble(("Vector database      ", "bold"), (report["sync_status"],
                                   "good" if report["sync_status"] == "synced" else "warn")))
    if report["changes"]:
        parts.append(Text(""))
        parts += [Text.assemble((f"  {kind:<9}", kind), name) for name, kind in report["changes"].items()]
    if report["untracked"]:
        parts.append(Text(""))
        parts += [Text.assemble(("  untracked", "muted"), f" {name}") for name in report["untracked"]]
    notes = []
    if report["config_changed"]:
        notes.append("Staging configuration changed since add.")
    if report["stale"]:
        notes.append("Embedding configuration changed; matching text has stale vectors.")
    if report["issues"]:
        notes.append(f"Missing or corrupt object references: {report['issues']} unique chunks need regeneration.")
    if report["checkout_incomplete"]:
        notes.append("Checkout publication is incomplete; /checkout HEAD recovers it.")
    if notes:
        parts.append(Text(""))
        parts += [Text(note, style="warn") for note in notes]
    step = _next_step(report)
    if step:
        parts += [Text(""), Text.assemble(("Next  ", "muted"), (step, "accent"))]
    elif not notes:
        parts += [Text(""), Text("Everything is committed and up to date.", style="good")]
    return _panel(Group(*parts), "Status")


def embedded(result) -> Panel:
    fresh = result.occurrences - result.reused
    grid = Table.grid(padding=(0, 2))
    grid.add_row(Text("Reused", style="bold"), share_bar(result.reused, result.occurrences),
                 Text(f"{result.reused} of {result.occurrences} occurrences"))
    grid.add_row(Text("Embedded", style="bold"), share_bar(fresh, result.occurrences, style="warn"),
                 Text(f"{result.generated} new objects"))
    return _panel(Group(grid, Text(""), Text.assemble((f"Finished in {result.elapsed_seconds:.2f}s", "muted"),
                                                      ("   Next  ", "muted"), ("/commit <message>", "accent"))), "Embeddings ready")


def history(entries: list[dict]) -> Panel:
    if not entries:
        return _panel(Text("No commits yet. Run /add, /embed, then /commit.", style="muted"), "History")
    lines = []
    for position, entry in enumerate(entries):
        head = Text.assemble(("● ", "accent"), (entry["short"], "sha"))
        for label in entry["labels"]:
            head.append("  ")
            head.append(f" {label} ", style=f"bold #1f1d1a on {ORANGE if label.startswith('HEAD') else MINT}")
        subject = entry["message"].splitlines()[0]
        older = entries[position + 1] if position + 1 < len(entries) else None
        lines += [head, Text.assemble(("│ ", "accent"), (subject, "bold")),
                  Text.assemble(("│ ", "accent"), (f"{entry['date'][:16].replace('T', ' ')} UTC · {entry['documents']} documents · "
                                                   f"{entry['chunks']} chunks", "muted"))]
        if older and older["commit"] == entry["parent"] and entry["vectors"]:
            shared = len(entry["vectors"] & older["vectors"])
            lines.append(Text.assemble(("│ ", "accent"), share_bar(shared, len(entry["vectors"]), width=20),
                                       (f" {shared} of {len(entry['vectors'])} vectors reused from parent", "muted")))
        if older:
            lines.append(Text("│", style="accent"))
    return _panel(Group(*lines), "History")


def _drift(pairs: list[dict]) -> Table | Text:
    # Identical text shares one vector, so only rewritten chunks can show drift.
    rewritten = [pair for pair in pairs if pair["match"] != "content" and "drift" in pair]
    if not rewritten:
        return Text("No rewritten chunks to measure.", style="muted")
    rewritten.sort(key=lambda pair: pair["drift"], reverse=True)
    table = Table(box=None, pad_edge=False, header_style="muted")
    table.add_column("Chunk")
    table.add_column("Cosine", justify="right")
    table.add_column("Drift", justify="right")
    for pair in rewritten[:DRIFT_ROWS]:
        table.add_row(Text(f"{pair['document']} #{pair['new_ordinal']}", overflow="fold"),
                      f"{pair['cosine_similarity']:.4f}", Text(f"{pair['drift']:.4f}", style="warn"))
    if len(rewritten) > DRIFT_ROWS:
        table.add_row(Text(f"and {len(rewritten) - DRIFT_ROWS} more", style="muted"), "", "")
    return table


def difference(result: dict) -> Panel:
    objects = result["embedding_objects"]
    kept = objects["shared"] + objects["new"]
    storage = result["storage"]
    parts = [_title(f"{result['old'][:7]} → {result['new'][:7]}"), Text(""),
             _counts({"Documents": result["documents"], "Chunks": result["chunks"]}), Text(""),
             Text.assemble(("Vectors   ", "bold"), share_bar(objects["shared"], kept),
                           f" {objects['shared']} shared, {objects['new']} new, {objects['removed']} removed"),
             Text.assemble(("Storage   ", "bold"), f"{bytes_label(storage['old_bytes'])} → {bytes_label(storage['new_bytes'])}",
                           (f"   {bytes_label(storage['additional_bytes'])} added on disk", "muted"))]
    notes = [text for flag, text in ((result["config_changed"], "Configuration changed between these snapshots."),
                                     (result["embedding_config_changed"], "Effective embedding configuration changed."))
             if flag]
    if notes:
        parts += [Text(""), *[Text(note, style="warn") for note in notes]]
    if result["embedding_config_changed"]:
        parts += [Text(""), Text("Vector drift is unavailable across embedding configurations.", style="muted")]
    else:
        parts += [Text(""), Text("Vector drift, largest first", style="bold"), _drift(result["pairs"]), Text(""),
                  Text("Chunks are paired by position, which is a heuristic rather than a semantic match.", style="muted")]
    return _panel(Group(*parts), "Diff")


def benchmark_rows(results: dict) -> dict[str, list[dict]]:
    """Average successful measured runs per scenario and strategy."""
    scenarios = {}
    for record in results.get("runs", []):
        scenarios.setdefault(record["scenario"], []).append(record)
    rows = {}
    for name, records in scenarios.items():
        group = {strategy: [record for record in records if record["strategy"] == strategy] for strategy in STRATEGIES}
        if not group["full"]:
            continue
        full = fmean(record["seconds"] for record in group["full"])
        rows[name] = []
        for strategy in STRATEGIES:
            if not group[strategy]:
                continue
            seconds = fmean(record["seconds"] for record in group[strategy])
            checks = [record["equivalence_to_full"]["equivalent"] for record in group[strategy]
                      if record.get("equivalence_to_full")]
            rows[name].append({"strategy": strategy, "seconds": seconds, "speedup": full / seconds if seconds else None,
                               "model_seconds": fmean(record["embedding_seconds"] for record in group[strategy]),
                               "inputs": fmean(record["model_inputs"] for record in group[strategy]),
                               "occurrences": fmean(record["occurrences"] for record in group[strategy]),
                               "equivalent": all(checks) if checks else None})
    return rows


def benchmark(results: dict, settings: dict) -> Panel:
    rows = benchmark_rows(results)
    if not rows:
        return _panel(Text("This run has no completed scenarios.", style="warn"), "Benchmark", style="warn")
    parts = [_title(f"Run {results['run_id']}"),
             Text(f"{settings.get('documents', '?')} synthetic documents · seed {settings.get('seed', '?')} · "
                  f"{settings.get('repeats', '?')} repeat(s) · values are means", style="muted"),
             Text("Bars show texts sent to the model. Total time also covers hashing, verification and commit.",
                  style="muted")]
    for name, group in rows.items():
        most = max(row["inputs"] for row in group)
        table = Table.grid(padding=(0, 2))
        table.add_column(no_wrap=True, min_width=19)
        table.add_column(no_wrap=True)
        table.add_column(justify="right", no_wrap=True)
        table.add_column(justify="right", no_wrap=True)
        table.add_column(justify="right", no_wrap=True)
        table.add_column(no_wrap=True)
        for row in group:
            ours = row["strategy"] == "chunk"
            table.add_row(Text(STRATEGY_LABELS[row["strategy"]], style="bold" if ours else ""),
                          share_bar(row["inputs"], most, style="good" if ours else "muted", width=14),
                          f"{row['inputs']:.0f} inputs", f"{row['model_seconds']:.2f}s model",
                          f"{row['seconds']:.2f}s total", _comparison(row))
        parts += [Text(""), Text.assemble((name, "bold"), _equivalence(group)), table]
    return _panel(Group(*parts), "Benchmark")


def _equivalence(group: list[dict]) -> tuple[str, str]:
    checks = [row["equivalent"] for row in group if row["equivalent"] is not None]
    if not checks:
        return "", ""
    return ("  all strategies produced the same vectors", "muted") if all(checks) else ("  DIFFERENT vectors", "bad")


def _comparison(row: dict) -> Text:
    if row["strategy"] == "full":
        return Text("baseline", style="muted")
    speedup = row["speedup"]
    if not speedup:
        return Text("n/a", style="muted")
    return Text(f"{speedup:.1f}× faster", style="good") if speedup >= 1 else Text(f"{1 / speedup:.1f}× slower", style="warn")
