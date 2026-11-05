"""The interactive shell: slash commands over the same parser and commands as the CLI."""
import contextlib
import difflib
import io
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import getpass
import time

from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from . import views
from .prompt import InputBox
from .. import objects, revisions, staging_config
from ..cli import ERRORS, build_parser
from ..commands.add import add
from ..commands.benchmark import benchmark
from ..commands.branch import branch
from ..commands.checkout import checkout
from ..commands.commit import commit
from ..commands.config import get as config_get
from ..commands.config import set_ as config_set
from ..commands.config import show as config_show
from ..commands.diff import compare
from ..commands.embed import embed
from ..commands.log import graph, history
from ..commands.status import inspect
from ..repository import initialize

DEFAULT_LOG_LIMIT = 10
FRAMES_PER_SECOND = 8


@dataclass(frozen=True)
class Command:
    name: str
    usage: str
    summary: str
    group: str


COMMANDS = (
    Command("add", "/add <paths>", "Track documents and stage their chunks", "Workflow"),
    Command("status", "/status", "Show changes and embedding readiness", "Workflow"),
    Command("embed", "/embed", "Generate only the embeddings that are missing", "Workflow"),
    Command("commit", "/commit <message>", "Save a snapshot and synchronize Chroma", "Workflow"),
    Command("tree", "/tree", "Show every branch and where you are", "History"),
    Command("log", "/log [count]", "Show committed history", "History"),
    Command("diff", "/diff [old] [new]", "Compare snapshots (default HEAD~1 to HEAD)", "History"),
    Command("branch", "/branch [name] [start]", "List or create branches", "History"),
    Command("checkout", "/checkout <revision> [--force]", "Restore a snapshot", "History"),
    Command("benchmark", "/benchmark [options]", "Measure regeneration strategies", "Evaluate"),
    Command("report", "/report", "Show the latest benchmark results again", "Evaluate"),
    Command("init", "/init [directory]", "Create a repository", "Session"),
    Command("config", "/config [show|get|set]", "Read or update configuration", "Session"),
    Command("help", "/help", "Show this list", "Session"),
    Command("clear", "/clear", "Clear the screen", "Session"),
    Command("exit", "/exit", "Leave the shell", "Session"),
)
NAMES = {command.name: command for command in COMMANDS}
ALIASES = {"quit": "exit"}


def repository_summary(directory: Path | str = ".") -> dict | None:
    """Cheap facts for the header and toolbar: no source scan, no model."""
    try:
        root = objects.find_repository(Path(directory))
        name, commit_id = revisions.head(root)
        documents = objects.read_index(root)["documents"]
    except (objects.RepositoryError, OSError, ValueError, KeyError):
        return None
    try:
        model = staging_config.load(root).embedding.get("model")
    except (staging_config.ConfigError, OSError, ValueError):
        model = None
    return {"name": root.name, "branch": name, "commit": commit_id, "model": model, "documents": len(documents),
            "chunks": sum(len(document.get("occurrences", [])) for document in documents.values())}


def commit_graph(directory: Path | str = ".") -> dict | None:
    try:
        return graph(directory)
    except (objects.RepositoryError, OSError):
        return None


def split_line(line: str) -> list[str]:
    """Split like a shell, but leave Windows backslashes in paths alone."""
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split, lexer.commenters, lexer.escape = True, "", ""
    return list(lexer)


def arguments_for(name: str, tokens: list[str]) -> list[str]:
    """Expand the shell's shorthand into the arguments the shared parser expects."""
    if name == "commit" and tokens and tokens[0] not in ("-m", "--message", "-h", "--help"):
        return ["--message", " ".join(tokens)]
    if name == "log":
        if not tokens:
            return ["--limit", str(DEFAULT_LOG_LIMIT)]
        if len(tokens) == 1 and tokens[0].lstrip("-").isdigit():
            return ["--limit", tokens[0]]
    if name == "diff" and len(tokens) < 2 and not set(tokens) & {"-h", "--help"}:
        return (tokens or ["HEAD~1"]) + ["HEAD"]
    if name == "config" and not tokens:
        return ["show"]
    return tokens


def revision_names(directory: Path | str = ".") -> list[str]:
    try:
        heads = objects.find_repository(Path(directory)) / ".embeddingvc/refs/heads"
        return ["HEAD", "HEAD~1", *sorted(path.name for path in heads.iterdir() if path.name != ".lock")]
    except (objects.RepositoryError, OSError):
        return ["HEAD", "HEAD~1"]


class SlashCompleter(Completer):
    def __init__(self):
        self.paths = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return
        if " " not in text:
            for command in COMMANDS:
                if command.name.startswith(text[1:]):
                    yield Completion("/" + command.name, start_position=-len(text), display_meta=command.summary)
            return
        name = text[1:].split(" ", 1)[0]
        word = "" if text.endswith(" ") else text.rsplit(" ", 1)[1]
        if name == "add":
            yield from self.paths.get_completions(Document(word), complete_event)
        elif name in ("checkout", "diff"):
            for revision in revision_names():
                if revision.startswith(word):
                    yield Completion(revision, start_position=-len(word))
        elif name == "config":
            for action in ("show", "get", "set"):
                if action.startswith(word):
                    yield Completion(action, start_position=-len(word))


class Working(Progress):
    """Live progress with the owl animated beside it."""

    def get_renderables(self):
        yield views.working(int(time.monotonic() * FRAMES_PER_SECOND), self.make_tasks_table(self.tasks))


class Shell:
    def __init__(self, console: Console | None = None):
        self.console = console or Console(theme=views.THEME)
        self.parser = build_parser()
        self.running = True

    def welcome(self) -> None:
        try:
            user = getpass.getuser().split()[0]
        except (OSError, KeyError, ImportError, IndexError):
            user = ""
        self.console.print(views.welcome(repository_summary(), commit_graph(), user=user, path=str(Path.cwd()),
                                         width=self.console.width))

    def execute(self, line: str) -> None:
        """Run one input line; failures are shown and the shell keeps going."""
        line = line.strip()
        if not line:
            return
        try:
            tokens = split_line(line)
        except ValueError as exc:
            self.console.print(views.failure(f"Cannot read that input: {exc}"))
            return
        name = ALIASES.get(tokens[0].removeprefix("/"), tokens[0].removeprefix("/"))
        if name not in NAMES:
            guess = difflib.get_close_matches(name, NAMES, n=1)
            hint = f" Did you mean /{guess[0]}?" if guess else ""
            self.console.print(views.failure(f"Unknown command /{name}.{hint} Type /help to see them all."))
            return
        try:
            getattr(self, "run_" + name)(*self.parsed(name, tokens[1:]))
        except KeyboardInterrupt:
            self.console.print(" Cancelled.", style="warn")
        except ERRORS as exc:
            self.console.print(views.failure(str(exc)))
        except Exception as exc:  # One bad command must not end a live session.
            self.console.print(views.failure(f"{type(exc).__name__}: {exc}"))

    def parsed(self, name: str, tokens: list[str]) -> tuple:
        if name in ("report", "tree", "help", "clear", "exit"):
            return ()
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                return (self.parser.parse_args([name, *arguments_for(name, tokens)]),)
        except SystemExit as exc:
            usage = re.sub(r"embeddingvc (\w+)", r"/\1", output.getvalue().strip())
            if exc.code:
                self.console.print(views.failure(usage))
            else:
                self.console.print(usage, markup=False, highlight=False)
            return (None,)

    def progress(self) -> Progress:
        bar = BarColumn(bar_width=24, style="muted", complete_style="orange", finished_style="orange", pulse_style="orange")
        return Working(TextColumn("[chalk]{task.description}"), bar, TextColumn("[muted]{task.fields[count]}"),
                       TimeElapsedColumn(), console=self.console, transient=True, refresh_per_second=FRAMES_PER_SECOND)

    @contextlib.contextmanager
    def busy(self, description: str):
        with self.progress() as progress:
            progress.add_task(description, total=None, count="")
            yield

    def run_help(self) -> None:
        self.console.print(views.help_table(COMMANDS))

    def run_clear(self) -> None:
        self.console.clear()
        self.welcome()

    def run_exit(self) -> None:
        self.running = False

    def run_init(self, args) -> None:
        if args:
            root = initialize(args.directory, force=args.force)
            self.console.print(views.message(f"Repository initialized at {root}\nAdd documents to data/, then run /add data"))

    def run_add(self, args) -> None:
        if args:
            with self.console.status("Staging documents", spinner_style="orange"):
                result = add(args.paths)
            self.console.print(views.message(result.render(), title="Staged"))

    def run_status(self, args) -> None:
        if args:
            with self.console.status("Comparing sources with the last snapshot", spinner_style="orange"):
                report = inspect()
            self.console.print(views.status(report))

    def run_embed(self, args) -> None:
        if not args:
            return
        with self.progress() as progress:
            task = progress.add_task("Reading documents", total=None, count="")

            def report(stage, done, total):
                progress.update(task, description=stage, completed=done, total=total or None,
                                count=f"{done}/{total}" if total else "")
            result = embed(progress=report)
        self.console.print(views.embedded(result))

    def run_commit(self, args) -> None:
        if args:
            with self.busy("Saving snapshot and synchronizing Chroma"):
                result = commit(message=args.message)
            self.console.print(views.message(result.render(), title="Committed"))

    def run_tree(self) -> None:
        self.console.print(views.tree_panel(graph()))

    def run_log(self, args) -> None:
        if args:
            self.console.print(views.history(history(limit=args.limit)))

    def run_diff(self, args) -> None:
        if args:
            with self.console.status("Comparing snapshots", spinner_style="orange"):
                result = compare(args.old, args.new)
            self.console.print(views.difference(result))

    def run_branch(self, args) -> None:
        if args:
            self.console.print(views.message(branch(name=args.name, start=args.start), title="Branches"))

    def run_checkout(self, args) -> None:
        if args:
            with self.busy(f"Restoring {args.revision}"):
                result = checkout(args.revision, force=args.force)
            self.console.print(views.message(result.render(), title="Checked out"))

    def run_config(self, args) -> None:
        if not args:
            return
        if args.config_command == "show":
            text = config_show()
        elif args.config_command == "get":
            text = config_get(Path("."), args.key)
        else:
            text = config_set(Path("."), args.key, args.value)
        self.console.print(views.shell_text(text), markup=False)

    def run_benchmark(self, args) -> None:
        if not args:
            return
        with self.progress() as progress:
            task = progress.add_task("Preparing benchmark", total=None, count="")

            def report(text):
                match = re.match(r"\[(\d+)/(\d+)\] (.*)", text)
                if match:
                    done, total, label = match.groups()
                    progress.update(task, description=label, completed=int(done) - 1, total=int(total),
                                    count=f"{done}/{total}")
                else:
                    progress.update(task, description=text)
            result = benchmark(seed=args.seed, repeats=args.repeats, documents=args.documents, top_k=args.top_k,
                               scenarios=args.scenarios, model_change_config=args.model_change_config,
                               keep_workspaces=args.keep_workspaces, warmup=args.warmup, progress=report)
        self.show_report(result.directory)
        if result.failures or result.skipped:
            self.console.print(views.message(result.render(), style="warn", title="Notes"))

    def run_report(self) -> None:
        root = objects.find_repository(Path("."))
        runs = sorted(path.parent for path in (root / "reports/benchmarks").glob("*/raw_results.json"))
        if not runs:
            self.console.print(views.message("No benchmark results yet.\nRun /benchmark first.", style="warn", title="Benchmark"))
            return
        self.show_report(runs[-1])

    def show_report(self, directory: Path) -> None:
        results = json.loads((directory / "raw_results.json").read_text(encoding="utf-8"))
        settings = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self.console.print(views.benchmark(results, settings))


def run() -> None:
    shell = Shell()
    shell.welcome()
    box = InputBox(SlashCompleter(), {"/" + command.name for command in COMMANDS if "<" in command.usage})
    while shell.running:
        try:
            line = box.read(views.toolbar(repository_summary()))
        except EOFError:
            break
        if line.strip():
            shell.console.print(views.echo(line))
            shell.execute(line)
            shell.console.print()
