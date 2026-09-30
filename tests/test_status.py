import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embeddingvc.commands.add import add
from embeddingvc.commands.config import set_
from embeddingvc.commands.status import StatusError, status
from embeddingvc.hashing import hash_payload
from embeddingvc.objects import read_index, publish_index, object_path
from embeddingvc.repository import initialize
from embeddingvc.staging_config import load


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = initialize(Path(self.temp.name) / "repo").resolve()

    def write(self, name="a.txt", text="alpha beta"):
        path = self.root / "data" / name
        path.write_text(text, encoding="utf-8")
        return path

    def stage(self):
        add([str(self.root / "data")], directory=self.root)

    def baseline(self, index=None):
        index = index or read_index(self.root)
        digest = hash_payload(index)
        (self.root / ".embeddingvc/commits" / digest).write_text(json.dumps(index), encoding="utf-8")
        (self.root / ".embeddingvc/refs/heads/main").write_text(digest, encoding="utf-8")
        return digest

    def vector(self):
        index = read_index(self.root)
        occurrence = index["documents"]["data/a.txt"]["occurrences"][0]
        fp = load(self.root).embedding_fingerprint()
        value = {"chunk": occurrence["chunk"], "fingerprint": fp, "values": [0.2, 0.4]}
        digest = hash_payload(value)
        path = object_path(self.root, "vectors", digest)
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        occurrence["embedding"] = {"vector": digest, "fingerprint": fp}
        publish_index(self.root, index)
        return path

    def test_empty_and_untracked(self):
        self.assertIn("No document or embedding changes", status(self.root))
        self.write()
        report = status(self.root)
        self.assertIn("Untracked documents", report)
        self.assertIn("Embeddings required: 0", report)

    def test_post_add_edit_and_read_only(self):
        path = self.write()
        self.stage()
        path.write_text("changed text", encoding="utf-8")
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = status(self.root / "data")
        after = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertIn("Working state differs from index", report)
        self.assertIn("Embeddings required: 1 occurrences (1 unique", report)

    def test_duplicate_counts_and_deletion(self):
        self.write()
        self.write("b.txt")
        self.stage()
        self.assertIn("2 occurrences (1 unique", status(self.root))
        self.baseline()
        (self.root / "data/a.txt").unlink()
        self.assertIn("Documents: 0 added, 0 modified, 1 deleted", status(self.root))

    def test_ready_and_revision_stale(self):
        self.write()
        self.stage()
        self.vector()
        self.baseline()
        self.assertIn("Embeddings required: 0", status(self.root))
        set_(self.root, "model_revision", "different-revision")
        report = status(self.root)
        self.assertIn("1 stale", report)
        self.assertIn("Embeddings required: 1", report)

    def test_corrupt_vector_and_config_require_regeneration(self):
        self.write()
        self.stage()
        path = self.vector()
        path.write_text("{}", encoding="utf-8")
        self.assertIn("regeneration required", status(self.root))
        self.vector()
        index = read_index(self.root)
        object_path(self.root, "configs", index["config"]).unlink()
        self.assertIn("Embeddings required: 1", status(self.root))

    def test_detached_and_sync(self):
        self.write()
        self.stage()
        digest = self.baseline()
        (self.root / ".embeddingvc/HEAD").write_text(digest, encoding="utf-8")
        for state in ("pending", "failed"):
            (self.root / ".embeddingvc/sync.json").write_text(json.dumps({"status": state}), encoding="utf-8")
            report = status(self.root)
            self.assertIn("HEAD detached", report)
            self.assertIn("Database synchronization: " + state, report)

    def test_config_change_and_parser_failure(self):
        self.write()
        self.stage()
        set_(self.root, "chunk_size", "200")
        self.assertIn("Staging configuration changed", status(self.root))
        (self.root / "data/a.txt").write_bytes(b"\xff")
        with self.assertRaises(StatusError):
            status(self.root)

    def test_cli_and_outside_repository(self):
        result = subprocess.run([sys.executable, "-m", "embeddingvc", "status"], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("On branch main", result.stdout)
        result = subprocess.run([sys.executable, "-m", "embeddingvc", "status"], cwd=self.temp.name, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("init", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_corrupt_head_fails_without_writing(self):
        self.write()
        self.stage()
        digest = self.baseline()
        (self.root / ".embeddingvc/commits" / digest).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(StatusError, "corrupt"):
            status(self.root)

    def test_missing_chunk_is_not_ready(self):
        self.write()
        self.stage()
        self.vector()
        occurrence = read_index(self.root)["documents"]["data/a.txt"]["occurrences"][0]
        object_path(self.root, "chunks", occurrence["chunk"]).unlink()
        report = status(self.root)
        self.assertIn("Embeddings required: 1", report)
        self.assertIn("regeneration required", report)
