import contextlib
import functools
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embeddingvc.cli import app
from embeddingvc.commands.add import add
from embeddingvc.commands.commit import commit
from embeddingvc.commands.config import set_
from embeddingvc.commands.embed import embed
from embeddingvc.commands.log import history, log
from embeddingvc.repository import initialize
from history_fixtures import snapshot, state
from test_commit import Encoder, Store

try:
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.document import Document
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from rich.console import Console
    from embeddingvc.shell import app as shell_app, views
    from embeddingvc.shell.app import COMMANDS, Shell, SlashCompleter, arguments_for, repository_summary, split_line
except ModuleNotFoundError:
    Console = None


class Parsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if Console is None:
            raise unittest.SkipTest("tui extra is not installed")

    def test_windows_paths_and_quotes_survive(self):
        self.assertEqual(split_line(r'/add data\notes.txt'), ["/add", r"data\notes.txt"])
        self.assertEqual(split_line('/commit "two words"'), ["/commit", "two words"])

    def test_shorthand_expands_to_parser_arguments(self):
        self.assertEqual(arguments_for("commit", ["fix", "typo"]), ["--message", "fix typo"])
        self.assertEqual(arguments_for("commit", ["-m", "fix"]), ["-m", "fix"])
        self.assertEqual(arguments_for("log", ["5"]), ["--limit", "5"])
        self.assertEqual(arguments_for("log", []), ["--limit", "10"])
        self.assertEqual(arguments_for("diff", []), ["HEAD~1", "HEAD"])
        self.assertEqual(arguments_for("diff", ["main~2"]), ["main~2", "HEAD"])
        self.assertEqual(arguments_for("diff", ["a", "b"]), ["a", "b"])
        self.assertEqual(arguments_for("config", []), ["show"])

    def test_cli_hints_become_slash_commands(self):
        text = 'Next: embeddingvc commit -m "x"\nEdit embeddingvc.yaml'
        self.assertEqual(views.shell_text(text), 'Next: /commit -m "x"\nEdit embeddingvc.yaml')

    def test_completer_lists_commands_only_for_slash_input(self):
        completer = SlashCompleter()
        complete = lambda text: [c.text for c in completer.get_completions(Document(text), None)]
        self.assertEqual(complete("/"), ["/" + command.name for command in COMMANDS])
        self.assertEqual(complete("/st"), ["/status"])
        self.assertEqual(complete("st"), [])
        self.assertIn("get", complete("/config "))


class ShellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if Console is None:
            raise unittest.SkipTest("tui extra is not installed")

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = initialize(Path(temp.name) / "repo").resolve()
        self.output = io.StringIO()
        self.shell = Shell(Console(file=self.output, width=110, theme=views.THEME))
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(self.root)

    def run_line(self, line):
        self.output.seek(0)
        self.output.truncate()
        self.shell.execute(line)
        return self.output.getvalue()

    def workflow(self):
        set_(self.root, "model_revision", "a" * 40)
        (self.root / "data/a.txt").write_text("first text", encoding="utf-8")
        (self.root / "data/b.txt").write_text("second text", encoding="utf-8")

    def test_workflow_through_the_shell(self):
        self.workflow()
        with patch.object(shell_app, "embed", functools.partial(embed, encoder_factory=Encoder)), \
                patch.object(shell_app, "commit", functools.partial(commit, adapter_factory=Store)):
            self.assertIn("Staged data", self.run_line("/add data"))
            report = self.run_line("/status")
            self.assertIn("data/a.txt", report)
            self.assertIn("/embed", report)
            self.assertIn("2 new objects", self.run_line("/embed"))
            self.assertIn("/commit <message>", self.run_line("/status"))
            self.assertIn("Created commit", self.run_line('/commit "first snapshot"'))
            self.assertIn("first snapshot", self.run_line("/log"))
            self.assertIn("up to date", self.run_line("/status"))
            (self.root / "data/a.txt").write_text("changed text", encoding="utf-8")
            self.run_line("/embed")
            self.run_line("/commit second")
            self.assertIn("vectors reused from parent", self.run_line("/log"))
        self.assertEqual(repository_summary()["documents"], 2)

    def test_errors_keep_the_shell_alive(self):
        self.assertIn("Did you mean /status", self.run_line("/statsu"))
        self.assertIn("Error", self.run_line("/embed"))
        self.assertIn("Error", self.run_line("/checkout nothing"))
        self.assertIn("required", self.run_line("/commit"))
        self.assertIn("Error", self.run_line("/log zero"))
        self.assertIn("usage", self.run_line("/add --help"))
        self.assertIn("Error", self.run_line('/commit "unterminated'))
        with patch.object(shell_app, "inspect", side_effect=KeyError("boom")):
            self.assertIn("KeyError", self.run_line("/status"))
        with patch.object(shell_app, "inspect", side_effect=KeyboardInterrupt):
            self.assertIn("Cancelled", self.run_line("/status"))
        self.assertTrue(self.shell.running)

    def test_exit_and_plain_words(self):
        self.assertIn("/commit <message>", self.run_line("help"))
        self.shell.execute("/quit")
        self.assertFalse(self.shell.running)

    def test_outside_a_repository(self):
        with tempfile.TemporaryDirectory() as empty, contextlib.chdir(empty):
            self.assertIsNone(repository_summary())
            self.assertIn("No repository", self.run_line("/clear"))
            self.assertIn("/init", views.toolbar(None))
            self.assertIn("Error", self.run_line("/status"))

    def test_user_text_is_not_treated_as_markup(self):
        snapshot(self.root, message="[bold]literal[/bold] [red")
        self.assertIn("[bold]literal[/bold] [red", self.run_line("/log"))

    def test_read_views_do_not_write(self):
        first, _ = snapshot(self.root, {"data/a.txt": [("old", [1.0, 0.0])]}, message="one")
        snapshot(self.root, {"data/a.txt": [("new", [0.0, 1.0])]}, parent=first, message="two")
        before = state(self.root)
        output = self.run_line("/diff")
        self.assertIn("Vector drift", output)
        self.assertIn("data/a.txt #0", output)
        self.assertIn("1.0000", output)
        self.run_line("/log")
        self.run_line("/branch")
        self.assertEqual(state(self.root), before)

    def test_report_reads_the_latest_run(self):
        self.assertIn("No benchmark results", self.run_line("/report"))
        record = {"scenario": "edit-5pct", "seconds": 4.0, "embedding_seconds": 3.0, "model_inputs": 100, "occurrences": 100}
        runs = [{**record, "strategy": "full", "equivalence_to_full": None},
                {**record, "strategy": "document", "seconds": 2.0, "model_inputs": 40,
                 "equivalence_to_full": {"equivalent": True}},
                {**record, "strategy": "chunk", "seconds": 1.0, "model_inputs": 5,
                 "equivalence_to_full": {"equivalent": True}}]
        for run_id, values in (("20260101T000000Z-aa", []), ("20260202T000000Z-bb", runs)):
            directory = self.root / "reports/benchmarks" / run_id
            directory.mkdir(parents=True)
            (directory / "raw_results.json").write_text(json.dumps({"run_id": run_id, "runs": values}), encoding="utf-8")
            (directory / "config.json").write_text(json.dumps({"documents": 20, "seed": 1, "repeats": 1}), encoding="utf-8")
        output = self.run_line("/report")
        self.assertIn("20260202T000000Z-bb", output)
        self.assertIn("4.0× faster", output)
        self.assertIn("same vectors", output)

    def test_report_says_slower_rather_than_a_fractional_speedup(self):
        row = {"strategy": "chunk", "speedup": 0.5}
        self.assertIn("2.0× slower", views._comparison(row).plain)


class Views(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if Console is None:
            raise unittest.SkipTest("tui extra is not installed")

    def test_share_bar_keeps_width_and_shows_small_values(self):
        self.assertEqual(views.share_bar(1, 1000, width=10).plain, "█" + "░" * 9)
        self.assertEqual(views.share_bar(0, 0, width=10).plain, "░" * 10)
        self.assertEqual(views.share_bar(5, 5, width=10).plain, "█" * 10)

    def test_bytes_label(self):
        self.assertEqual(views.bytes_label(12), "12 B")
        self.assertEqual(views.bytes_label(2048), "2.0 KB")
        self.assertEqual(views.bytes_label(3 * 1024 ** 3), "3.0 GB")

    def test_benchmark_rows_skip_runs_without_a_full_baseline(self):
        self.assertEqual(views.benchmark_rows({"runs": [{"scenario": "x", "strategy": "chunk"}]}), {})


class HistoryData(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = initialize(Path(temp.name) / "repo")

    def test_entries_back_the_plain_log(self):
        self.assertEqual(history(self.root), [])
        first, _ = snapshot(self.root, {"a.txt": [("text", [1.0, 0.0])]}, message="one")
        snapshot(self.root, {"a.txt": [("text", [1.0, 0.0])]}, parent=first, message="two")
        newest, oldest = history(self.root)
        self.assertEqual((newest["message"], oldest["message"]), ("two", "one"))
        self.assertEqual(newest["vectors"], oldest["vectors"])
        self.assertEqual(oldest["parent_short"], None)
        self.assertIn("two", log(self.root, limit=1))
        self.assertNotIn("one", log(self.root, limit=1))


class EmbedProgress(unittest.TestCase):
    def test_batches_are_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = initialize(Path(temp) / "repo")
            set_(root, "model_revision", "a" * 40)
            (root / "data/a.txt").write_text("first text", encoding="utf-8")
            add([str(root / "data")], directory=root)
            events = []
            embed(root, encoder_factory=Encoder, progress=lambda *event: events.append(event))
            self.assertEqual(events[0], ("Loading model", 0, 0))
            self.assertEqual(events[-1], ("Embedding chunks", 1, 1))


class PromptLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if Console is None:
            raise unittest.SkipTest("tui extra is not installed")

    def typed(self, keys):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp, contextlib.chdir(temp), contextlib.redirect_stdout(output), \
                create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text(keys)
            shell_app.run()
        return output.getvalue()

    def test_commands_run_until_exit(self):
        output = self.typed("/help\r/exit\r/status\r")
        self.assertIn("No repository", output)
        self.assertIn("/commit <message>", output)
        self.assertNotIn("Not an EmbeddingVC", output)

    def test_ctrl_c_keeps_the_prompt_and_ctrl_d_leaves(self):
        self.assertIn("/commit <message>", self.typed("half typed\x03/help\r\x04"))


class Entry(unittest.TestCase):
    def test_no_arguments_outside_a_terminal_stays_a_usage_error(self):
        with patch.object(sys, "argv", ["embeddingvc"]), contextlib.redirect_stderr(io.StringIO()), \
                patch("sys.stdin", io.StringIO()), self.assertRaises(SystemExit) as raised:
            app()
        self.assertEqual(raised.exception.code, 2)

    def test_shell_outside_a_terminal_explains_itself(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), patch("sys.stdin", io.StringIO()), self.assertRaises(SystemExit) as raised:
            app(["shell"])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("interactive", stderr.getvalue())

    def test_missing_extra_names_the_install_command(self):
        stderr = io.StringIO()
        with patch.dict(sys.modules, {"prompt_toolkit": None}), contextlib.redirect_stderr(stderr):
            sys.modules.pop("embeddingvc.shell.app", None)
            with self.assertRaises(SystemExit):
                app(["shell"])
        self.assertIn("tui", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
