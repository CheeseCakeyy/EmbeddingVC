import contextlib
import io
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embeddingvc import commit_manager as manager, objects
from embeddingvc.cli import app
from embeddingvc.commands.add import add
from embeddingvc.commands.branch import branch, BranchError
from embeddingvc.commands.commit import commit, CommitError
from embeddingvc.commands.config import set_
from embeddingvc.commands.embed import embed
from embeddingvc.commands.status import status
from embeddingvc.hashing import hash_payload
from embeddingvc.object_store import read_object, write_object
from embeddingvc.repository import initialize, repository_lock


class Encoder:
    dimension = 3
    provenance = {"provider": "test", "revision": "a" * 40}

    def __init__(self, settings):
        pass

    def encode(self, texts, **kwargs):
        return [[1.0, 0.0, 0.0] for text in texts]


class Store:
    calls = []

    def __init__(self, root, settings):
        pass

    def replace(self, digest, rows, dimension):
        self.calls.append((digest, rows, dimension))
        return "verified-test-collection"


class CommitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = initialize(Path(self.temp.name) / "repo").resolve()
        self.meta = self.root / ".embeddingvc"
        set_(self.root, "model_revision", "a" * 40)
        self.file = self.root / "data/a.txt"
        self.file.write_text("alpha beta", encoding="utf-8")
        add([str(self.root / "data")], directory=self.root)
        self.embed()
        Store.calls = []

    def embed(self):
        return embed(self.root, encoder_factory=Encoder)

    def commit(self, **kwargs):
        return commit(self.root, message="test snapshot", adapter_factory=Store, **kwargs)

    def commits(self):
        return list((self.meta / "commits").glob("*.json"))

    def sync(self):
        return json.loads((self.meta / "sync.json").read_text())

    def assert_blocked(self, text=None):
        before = objects.index_path(self.root).read_bytes()
        head = (self.meta / "refs/heads/main").read_bytes()
        count = len(self.commits())
        with self.assertRaises(CommitError) as raised:
            self.commit()
        if text:
            self.assertIn(text, str(raised.exception))
        self.assertEqual(objects.index_path(self.root).read_bytes(), before)
        self.assertEqual((self.meta / "refs/heads/main").read_bytes(), head)
        self.assertEqual(len(self.commits()), count)

    def test_parentage_complete_snapshots_and_shared_objects(self):
        (self.root / "data/b.txt").write_text("alpha beta", encoding="utf-8")
        self.embed()
        first = self.commit()
        manifest = manager.read_commit(self.root, first.commit)
        self.assertIsNone(manifest["parent"])
        self.assertEqual(hash_payload(manifest), first.commit)
        self.assertEqual(manifest["statistics"], {"documents": 2, "occurrences": 2, "unique_vectors": 1})
        self.assertEqual(objects.read_index(self.root)["base_commit"], first.commit)
        self.assertEqual(len(list((self.meta / "objects/embeddings").glob("*.json"))), 1)
        self.assertEqual(len({row["id"] for row in Store.calls[-1][1]}), 2)
        self.assertIn("On branch main", status(self.root))
        branch(self.root, "experiment", first.commit[:12])
        self.assertEqual((self.meta / "refs/heads/experiment").read_text().strip(), first.commit)
        self.file.write_text("new text", encoding="utf-8")
        self.embed()
        second = self.commit()
        latest = manager.read_commit(self.root, second.commit)
        self.assertEqual(latest["parent"], first.commit)
        self.assertEqual(set(latest["documents"]), {"data/a.txt", "data/b.txt"})
        self.assertEqual(manager.read_commit(self.root, first.commit), manifest)

    def test_noop_even_after_embed(self):
        self.commit()
        self.embed()
        self.assert_blocked("Nothing to commit")

    def test_deletion_only_and_empty_snapshot(self):
        (self.root / "data/b.txt").write_text("second", encoding="utf-8")
        self.embed()
        self.commit()
        self.file.unlink()
        self.embed()
        second = self.commit()
        self.assertEqual(second.documents, 1)
        (self.root / "data/b.txt").unlink()
        self.embed()
        third = self.commit()
        self.assertEqual((third.documents, third.occurrences), (0, 0))
        self.assertEqual(Store.calls[-1][1], [])
        self.assertEqual(self.sync()["records"], 0)

    def test_source_change_new_file_and_deletion_block(self):
        self.file.write_text("changed", encoding="utf-8")
        self.assert_blocked("embed")
        self.embed()
        new = self.root / "data/new.txt"
        new.write_text("new", encoding="utf-8")
        self.assert_blocked("embed")
        new.unlink()
        self.file.unlink()
        self.assert_blocked("embed")

    def test_config_change_blocks(self):
        set_(self.root, "chunk_size", "900")
        self.assert_blocked("embed")

    def test_base_and_detached_head_block(self):
        index = objects.read_index(self.root)
        index["base_commit"] = "f" * 64
        objects.publish_index(self.root, index)
        self.assert_blocked("base")
        (self.meta / "HEAD").write_text("a" * 64)
        self.assert_blocked("Detached")

    def test_message_required_and_cli(self):
        for message in ("", "  ", None):
            with self.assertRaises(CommitError):
                commit(self.root, message=message, adapter_factory=Store)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            app(["commit"])
        self.assertEqual(raised.exception.code, 2)
        with patch("embeddingvc.cli.commit", side_effect=CommitError("test failure")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                app(["commit", "-m", "test"])
        self.assertEqual(raised.exception.code, 1)

    def test_missing_and_corrupt_vector_blocks(self):
        index = objects.read_index(self.root)
        ref = index["documents"]["data/a.txt"]["occurrences"][0]["embedding"]["vector"]
        path = objects.object_path(self.root, "embeddings", ref)
        raw = path.read_bytes()
        path.unlink()
        self.assert_blocked("Missing/corrupt")
        path.write_text("{}")
        self.assert_blocked("Missing/corrupt")
        path.write_bytes(raw)
        self.commit()

    def test_valid_but_incompatible_vector_blocks(self):
        index = objects.read_index(self.root)
        occurrence = index["documents"]["data/a.txt"]["occurrences"][0]
        vector = read_object(self.root, "embeddings", occurrence["embedding"]["vector"])
        vector["provenance"] = {"provider": "other"}
        occurrence["embedding"]["vector"] = write_object(self.root, "embeddings", vector)
        objects.publish_index(self.root, index)
        self.assert_blocked("mismatch")

    def test_tampered_occurrence_blocks(self):
        index = objects.read_index(self.root)
        index["documents"]["data/a.txt"]["occurrences"][0]["id"] = "forged"
        objects.publish_index(self.root, index)
        self.assert_blocked("stale")

    def test_sync_failure_is_durable_and_retry_creates_no_commit(self):
        with patch.object(Store, "replace", side_effect=RuntimeError("database unavailable")):
            with self.assertRaisesRegex(CommitError, "created; vector-store sync failed.*checkout HEAD"):
                self.commit()
        digest = objects.read_index(self.root)["base_commit"]
        self.assertEqual(self.sync()["status"], "failed")
        self.assertEqual((self.meta / "refs/heads/main").read_text().strip(), digest)
        # A retry repairs precisely that snapshot, even after source edits.
        self.file.write_text("not yet embedded", encoding="utf-8")
        result = self.commit()
        self.assertTrue(result.recovered)
        self.assertEqual(result.commit, digest)
        self.assertEqual(len(self.commits()), 1)
        self.assertEqual(Store.calls[-1][1][0]["text"], "alpha beta")
        self.assertEqual(self.sync()["status"], "synced")

    def test_every_publication_write_can_be_recovered(self):
        original = objects._atomic_write
        targets = ("refs/heads/main", "index.json", "sync.json")
        for suffix in targets:
            with self.subTest(suffix=suffix):
                self.file.write_text(f"new {suffix}", encoding="utf-8")
                self.embed()
                def fail(path, payload):
                    if path.as_posix().endswith(suffix):
                        raise OSError("injected publication failure")
                    return original(path, payload)
                with patch.object(objects, "_atomic_write", side_effect=fail):
                    with self.assertRaisesRegex(CommitError, "publication interrupted"):
                        self.commit()
                self.assertTrue((self.meta / "transactions/commit.json").exists())
                count = len(self.commits())
                result = self.commit()
                self.assertTrue(result.recovered)
                self.assertEqual(len(self.commits()), count)
                self.assertEqual(objects.read_index(self.root)["base_commit"], result.commit)
                self.assertFalse((self.meta / "transactions/commit.json").exists())
                self.assertEqual(self.sync()["status"], "synced")

    def test_recovery_before_other_mutations(self):
        with patch.object(objects, "publish_index", side_effect=OSError("crash")):
            with self.assertRaises(CommitError):
                self.commit()
        # config mutation must finish the index update first.
        set_(self.root, "chunk_size", "900")
        self.assertEqual(objects.read_index(self.root)["base_commit"], self.commits()[0].stem)
        self.assertFalse((self.meta / "transactions/commit.json").exists())

    def test_marker_failure_does_not_advance_head(self):
        with patch.object(manager, "write_json", side_effect=OSError("no space")):
            with self.assertRaisesRegex(CommitError, "publication interrupted"):
                self.commit()
        self.assertEqual((self.meta / "refs/heads/main").read_text(), "")
        self.assertIsNone(objects.read_index(self.root)["base_commit"])

    def test_ready_marker_failure_never_reports_success(self):
        original = manager.write_json
        def fail(root, path, value):
            if value.get("status") == "synced":
                raise OSError("ready marker failed")
            return original(root, path, value)
        with patch.object(manager, "write_json", side_effect=fail):
            with self.assertRaisesRegex(CommitError, "sync failed"):
                self.commit()
        self.assertEqual(self.sync()["status"], "failed")
        self.assertTrue(self.commit().recovered)

    def test_shared_lock_blocks_commit_and_branch(self):
        self.commit()
        with repository_lock(self.root):
            self.assert_blocked("locked")
            with self.assertRaisesRegex(BranchError, "locked"):
                branch(self.root, "experiment")

    def test_sync_commit_without_sources(self):
        result = self.commit()
        self.file.unlink()
        with repository_lock(self.root):
            count = manager.sync_commit(self.root, result.commit, adapter_factory=Store)
        self.assertEqual(count, 1)

    def test_process_death_releases_lock_and_recovers_transaction(self):
        import subprocess
        import sys
        script = '''
import os, sys
from embeddingvc import objects
from embeddingvc.commands.commit import commit
def crash(*args, **kwargs):
    os._exit(23)
objects.publish_index = crash
commit(sys.argv[1], message="interrupted process")
'''
        result = subprocess.run([sys.executable, "-c", script, str(self.root)], capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertTrue((self.meta / "lock").exists())
        self.assertTrue((self.meta / "transactions/commit.json").exists())
        recovered = self.commit()
        self.assertTrue(recovered.recovered)
        self.assertEqual(len(self.commits()), 1)
        self.assertFalse((self.meta / "lock").exists())

    def test_recovery_refuses_conflicting_index_without_losing_marker(self):
        with patch.object(objects, "publish_index", side_effect=OSError("crash")):
            with self.assertRaises(CommitError):
                self.commit()
        index = objects.read_index(self.root)
        index["documents"] = {}
        objects.publish_index(self.root, index)
        with self.assertRaisesRegex(CommitError, "conflicts"):
            self.commit()
        self.assertTrue((self.meta / "transactions/commit.json").exists())

    def test_changed_sources_during_validation_are_rejected(self):
        original = manager.records
        def change(*args):
            result = original(*args)
            self.file.write_text("concurrent edit", encoding="utf-8")
            return result
        with patch.object(manager, "records", side_effect=change):
            self.assert_blocked("embed")

    def test_missing_chunk_and_config_references(self):
        index = objects.read_index(self.root)
        references = [("chunks", index["documents"]["data/a.txt"]["occurrences"][0]["chunk"]),
                      ("configs", index["config_object"]), ("configs", index["embedding_config"])]
        for kind, digest in references:
            path = objects.object_path(self.root, kind, digest)
            raw = path.read_bytes()
            path.unlink()
            self.assert_blocked("Missing/corrupt")
            path.write_bytes(raw)

    def test_unembedded_add_is_rejected(self):
        self.file.write_text("changed source", encoding="utf-8")
        add([str(self.root / "data")], directory=self.root)
        self.assert_blocked("not ready")

    def test_first_empty_candidate_rejected(self):
        self.file.unlink()
        self.embed()
        self.assert_blocked("Nothing to commit")

    def test_commit_does_not_load_encoder(self):
        with patch("embeddingvc.commands.embed.SentenceTransformerEncoder", side_effect=AssertionError("no encoder")):
            self.commit()

    @unittest.skipUnless(importlib.util.find_spec("chromadb"), "Install .[vector-store] for real Chroma integration")
    def test_real_cli_commit_noop_and_empty_replacement(self):
        import subprocess
        import sys
        def run():
            return subprocess.run([sys.executable, "-m", "embeddingvc", "commit", "-m", "CLI snapshot"],
                                  cwd=self.root, capture_output=True, text=True, timeout=90)
        first = run()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertIn("Created commit", first.stdout)
        self.assertIn("Chroma synchronized: 1 records", first.stdout)
        digest = self.sync()["commit"]
        unchanged = run()
        self.assertEqual(unchanged.returncode, 1, unchanged.stdout + unchanged.stderr)
        self.assertIn("Nothing to commit", unchanged.stderr)
        self.file.unlink()
        self.embed()
        empty = run()
        self.assertEqual(empty.returncode, 0, empty.stdout + empty.stderr)
        self.assertIn("Chroma synchronized: 0 records", empty.stdout)
        self.assertEqual(manager.read_commit(self.root, self.sync()["commit"])["parent"], digest)


if __name__ == "__main__":
    unittest.main()
