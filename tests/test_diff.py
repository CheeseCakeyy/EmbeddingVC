import copy
import math
import tempfile
import unittest
from pathlib import Path
from embeddingvc.commands.diff import diff, DiffError
from embeddingvc.diff_engine import compare_snapshots, vector_metrics
from embeddingvc.repository import initialize
from history_fixtures import snapshot, state


class DiffTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = initialize(Path(temp.name) / "repo")

    def test_identical_and_storage_deduplication(self):
        digest, old = snapshot(self.root, {"a": [("same", [1, 0]), ("same", [1, 0])]})
        before = state(self.root)
        result = compare_snapshots(self.root, old, old)
        self.assertEqual(result["chunks"]["unchanged"], 2)
        self.assertEqual(result["embedding_objects"], {"shared": 1, "new": 0, "removed": 0})
        self.assertEqual(result["storage"]["additional_bytes"], 0)
        self.assertEqual(result["storage"]["old_bytes"], sum(p.stat().st_size for p in (self.root / ".embeddingvc/objects").rglob("*.json")))
        self.assertIn("drift=0", diff(digest, "HEAD", self.root))
        self.assertEqual(state(self.root), before)

    def test_content_first_and_ordinal_heuristic(self):
        _, old = snapshot(self.root, {"a": [("alpha", [1, 0]), ("beta", [1, 0])]})
        _, new = snapshot(self.root, {"a": [("inserted", [0, 1]), ("alpha", [1, 0]), ("beta", [1, 0])]})
        result = compare_snapshots(self.root, old, new)
        self.assertEqual(result["chunks"]["unchanged"], 2)
        self.assertEqual(result["chunks"]["added"], 1)
        _, new = snapshot(self.root, {"a": [("changed", [0, 1]), ("beta", [1, 0])]})
        result = compare_snapshots(self.root, old, new)
        pair = next(p for p in result["pairs"] if p["match"] == "ordinal heuristic")
        self.assertEqual(pair["drift"], 1)
        self.assertAlmostEqual(pair["euclidean_distance"], math.sqrt(2))

    def test_model_changes_and_metadata(self):
        _, old = snapshot(self.root, {"a": [("alpha", [1, 0])]})
        _, new = snapshot(self.root, {"a": [("alpha", [1, 0])]}, model="b")
        result = compare_snapshots(self.root, old, new)
        self.assertIn("incompatible", result["pairs"][0]["unavailable"])
        self.assertEqual(result["chunks"]["stale"], 1)
        new = copy.deepcopy(old)
        new["documents"]["a"]["metadata"] = {"title": "new"}
        result = compare_snapshots(self.root, old, new)
        self.assertEqual(result["documents"]["metadata_changed"], 1)
        self.assertEqual(result["chunks"]["modified"], 0)

    def test_add_delete_empty_and_bad_objects(self):
        first, old = snapshot(self.root, {"a": [("alpha", [1, 0])]})
        _, new = snapshot(self.root, {"b": [("beta", [1, 0])]}, parent=first)
        result = compare_snapshots(self.root, old, new)
        self.assertEqual(result["chunks"]["added"], 1)
        self.assertEqual(result["chunks"]["deleted"], 1)
        self.assertIn("no paired chunks", diff("HEAD~1", "HEAD", self.root))
        reference = old["documents"]["a"]["occurrences"][0]["embedding"]["vector"]
        (self.root / f".embeddingvc/objects/embeddings/{reference}.json").write_text('{"vector": [NaN]}')
        before = state(self.root)
        with self.assertRaisesRegex(DiffError, "corrupt"):
            diff(first, "HEAD", self.root)
        self.assertEqual(state(self.root), before)

    def test_zero_and_large_vectors(self):
        self.assertIn("zero-norm", vector_metrics([0, 0], [1, 0])["unavailable"])
        self.assertAlmostEqual(vector_metrics([1e300, 1e300], [1e300, 1e300])["cosine_similarity"], 1)
        self.assertEqual(vector_metrics([1, 0], [-1, 0])["drift"], 2)

    def test_real_commit_workflow_and_cli_read_only(self):
        import contextlib
        import io
        from unittest.mock import patch
        from embeddingvc.cli import app
        from embeddingvc.commands.add import add
        from embeddingvc.commands.config import set_
        from embeddingvc.commands.embed import embed
        from embeddingvc.commands.commit import commit
        from test_commit import Encoder, Store
        set_(self.root, "model_revision", "a" * 40)
        source = self.root / "data/notes.txt"
        source.write_text("first text")
        add([str(source)], directory=self.root)
        embed(self.root, encoder_factory=Encoder)
        commit(self.root, message="first", adapter_factory=Store)
        source.write_text("second text")
        embed(self.root, encoder_factory=Encoder)
        commit(self.root, message="second", adapter_factory=Store)
        source.unlink()
        (self.root / "embeddingvc.yaml").write_text("broken working config")
        before = state(self.root)
        with contextlib.chdir(self.root), patch.object(Encoder, "encode", side_effect=AssertionError("model called")), patch.object(Store, "replace", side_effect=AssertionError("store called")):
            for args, expected in ((["log"], "second"), (["diff", "HEAD~1", "HEAD"], "1 modified")):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    app(args)
                self.assertIn(expected, output.getvalue())
        self.assertEqual(state(self.root), before)
