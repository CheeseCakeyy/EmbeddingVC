import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from embeddingvc.cli import app
from embeddingvc.commands.log import log, LogError
from embeddingvc.repository import initialize
from embeddingvc.revisions import resolve_revision, short_id
from history_fixtures import snapshot, state


class LogTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = initialize(Path(temp.name) / "repo")

    def test_empty_and_invalid_limit(self):
        self.assertEqual(log(self.root), "No commits yet")
        for limit in (0, -1, True, "2"):
            with self.assertRaises(LogError):
                log(self.root, limit=limit)
        for value in ("0", "-1", "bad"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                app(["log", "--limit", value])
            self.assertEqual(exc.exception.code, 2)

    def test_chain_limit_branch_detached_and_read_only(self):
        first, _ = snapshot(self.root, message="first")
        second, _ = snapshot(self.root, parent=first, message="second")
        (self.root / ".embeddingvc/refs/heads/experiment").write_text(first)
        before = state(self.root)
        output = log(self.root)
        self.assertLess(output.index("second"), output.index("first"))
        self.assertIn("HEAD -> main", output)
        self.assertIn("experiment", output)
        self.assertNotIn("    first", log(self.root, limit=1))
        self.assertEqual(state(self.root), before)
        self.assertEqual(resolve_revision(self.root, "HEAD~1"), first)
        self.assertEqual(resolve_revision(self.root, second[:12]), second)
        self.assertEqual(resolve_revision(self.root, "experiment"), first)
        (self.root / ".embeddingvc/HEAD").write_text(first)
        self.assertIn("HEAD", log(self.root))
        self.assertNotIn("    second", log(self.root))
        (self.root / ".embeddingvc/HEAD").write_text("ref: refs/heads/experiment")
        self.assertNotIn("    second", log(self.root))

    def test_bad_parent_corruption_and_cycle(self):
        digest, manifest = snapshot(self.root, parent="f" * 64)
        before = state(self.root)
        with self.assertRaisesRegex(LogError, "Missing/corrupt"):
            log(self.root)
        self.assertEqual(state(self.root), before)
        manifest["parent"] = digest
        with patch("embeddingvc.revisions.manifest", return_value=manifest):
            with self.assertRaisesRegex(LogError, "Cycle"):
                log(self.root)
        (self.root / f".embeddingvc/commits/{digest}.json").write_text("{}")
        with self.assertRaisesRegex(LogError, "corrupt"):
            log(self.root)

    def test_revision_errors_and_unique_abbreviation(self):
        snapshot(self.root)
        with self.assertRaisesRegex(LogError, "ancestor"):
            resolve_revision(self.root, "HEAD~1")
        for rev in ("../main", "HEAD~-1", "unknown"):
            with self.assertRaises(LogError):
                resolve_revision(self.root, rev)
        ids = ["abcdefg1" + "a" * 56, "abcdefg2" + "b" * 56]
        self.assertEqual(short_id(ids[0], ids), "abcdefg1")
        with patch("embeddingvc.revisions.commit_ids", return_value=["a" * 64, "a" * 63 + "b"]):
            with self.assertRaisesRegex(LogError, "Ambiguous"):
                resolve_revision(self.root, "aaaa")
