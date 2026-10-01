import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embeddingvc.cli import app
from embeddingvc.commands.add import AddError, add
from embeddingvc.commands.config import set_
from embeddingvc.commands.embed import EmbedError, embed
from embeddingvc.commands.status import status
from embeddingvc.config import ConfigurationError
from embeddingvc.embedding_engine import EmbeddingError, SentenceTransformerEncoder
from embeddingvc.hashing import hash_text
from embeddingvc.object_store import read_object
from embeddingvc.objects import index_path, object_path, read_index
from embeddingvc.repository import initialize, repository_lock

REVISION = "a" * 40


class FakeEncoder:
    def __init__(self, settings, calls, callback=None):
        self.dimension = 3
        self.provenance = {"provider": "fake", "model": settings["model"],
                           "model_revision": settings["revision"], "tokenizer": "fake-v1"}
        self.calls = calls
        self.callback = callback

    def encode(self, texts, *, batch_size, normalize_embeddings):
        self.calls.append((list(texts), batch_size, normalize_embeddings))
        if self.callback:
            self.callback()
        vectors = []
        for text in texts:
            values = [int(hash_text(text)[i:i + 4], 16) / 65535 for i in (0, 4, 8)]
            if normalize_embeddings:
                norm = math.sqrt(sum(value * value for value in values))
                values = [value / norm for value in values]
            vectors.append(values)
        return vectors


class EmbedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = initialize(Path(self.temp.name) / "repo").resolve()
        set_(self.root, "model_revision", REVISION)
        self.calls = []
        self.callback = None

    def write(self, name="a.txt", text="alpha beta"):
        path = self.root / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def stage(self, path=None):
        add([str(path or self.root / "data")], directory=self.root)

    def factory(self, settings):
        return FakeEncoder(settings, self.calls, self.callback)

    def run_embed(self):
        return embed(self.root, encoder_factory=self.factory)

    def objects(self, kind="embeddings"):
        return list((self.root / ".embeddingvc/objects" / kind).glob("*.json"))

    def reference(self):
        return read_index(self.root)["documents"]["data/a.txt"]["occurrences"][0]["embedding"]

    def assert_preserved(self, action, message=None):
        before = index_path(self.root).read_bytes()
        with self.assertRaises(EmbedError) as raised:
            action()
        if message:
            self.assertIn(message, str(raised.exception))
        self.assertEqual(index_path(self.root).read_bytes(), before)
        self.assertFalse((self.root / ".embeddingvc/lock").exists())

    def test_no_tracked_inputs_explains_add(self):
        self.assert_preserved(self.run_embed, "embeddingvc add data")

    def test_unique_generation_duplicates_and_zero_encoder_calls_on_second_run(self):
        self.write()
        self.write("b.md")
        self.stage()
        first = self.run_embed()
        self.assertEqual((first.generated, first.reused, first.occurrences), (1, 1, 2))
        self.assertEqual(self.calls[0][0], ["alpha beta"])
        self.assertEqual(len(self.objects()), 1)
        refs = [doc["occurrences"][0]["embedding"] for doc in read_index(self.root)["documents"].values()]
        self.assertEqual(refs[0], refs[1])
        self.calls.clear()
        second = self.run_embed()
        self.assertEqual((second.generated, second.reused, second.occurrences), (0, 2, 2))
        self.assertEqual(self.calls, [])
        self.assertIn("0 new objects, 2 reused occurrences", second.render())
        self.assertIn("Elapsed:", second.render())

    def test_rescans_additions_edits_and_deletions_without_add(self):
        edited = self.write()
        deleted = self.write("b.txt", "will disappear")
        self.stage()
        self.run_embed()
        historical = set(self.objects())
        edited.write_text("changed alpha", encoding="utf-8")
        deleted.unlink()
        self.write("c.txt", "newly discovered")
        self.calls.clear()
        self.assertEqual(self.run_embed().generated, 2)
        self.assertEqual(set(read_index(self.root)["documents"]), {"data/a.txt", "data/c.txt"})
        self.assertTrue(historical <= set(self.objects()))

    def test_all_deleted_roots_publish_empty_candidate_and_keep_history(self):
        path = self.write()
        self.stage(path)
        self.run_embed()
        historical = set(self.objects())
        path.unlink()
        self.calls.clear()
        result = self.run_embed()
        self.assertEqual((result.generated, result.occurrences), (0, 0))
        self.assertEqual(read_index(self.root)["documents"], {})
        self.assertEqual(self.calls, [])
        self.assertEqual(set(self.objects()), historical)

    def test_empty_tracked_directory_is_ready(self):
        self.stage()
        self.assertEqual(self.run_embed().occurrences, 0)
        self.assertEqual(read_index(self.root)["generation"]["status"], "ready")
        self.assertEqual(self.calls, [])

    def test_only_tracked_roots_are_refreshed(self):
        tracked = self.write()
        self.stage(tracked)
        self.write("untracked.txt", "untracked")
        self.run_embed()
        self.assertEqual(list(read_index(self.root)["documents"]), ["data/a.txt"])

    def test_revision_change_invalidates_and_restore_reuses_historical_objects(self):
        self.write()
        self.stage()
        self.run_embed()
        old = self.reference()
        set_(self.root, "model_revision", "b" * 40)
        self.assertEqual(self.run_embed().generated, 1)
        self.assertNotEqual(old["vector"], self.reference()["vector"])
        set_(self.root, "model_revision", REVISION)
        self.calls.clear()
        self.assertEqual(self.run_embed().generated, 0)
        self.assertEqual(self.reference(), old)
        self.assertEqual(self.calls, [])

    def test_pipeline_changes_invalidate_even_identical_normalized_text(self):
        self.write(text="already lowercase")
        self.stage()
        self.run_embed()
        before = self.reference()["vector"]
        set_(self.root, "preprocessing.lowercase", "true")
        self.assertIn("Embeddings required: 1", status(self.root))
        self.assertEqual(self.run_embed().generated, 1)
        self.assertNotEqual(self.reference()["vector"], before)

    def test_batch_name_and_database_changes_do_not_invalidate_vectors(self):
        self.write()
        self.stage()
        self.run_embed()
        reference = self.reference()
        set_(self.root, "embedding.batch_size", "1")
        set_(self.root, "repository.name", "renamed")
        set_(self.root, "vector_store.persist_directory", "output/other")
        self.calls.clear()
        self.assertEqual(self.run_embed().generated, 0)
        self.assertEqual(self.reference(), reference)
        stored = read_object(self.root, "configs", read_index(self.root)["config_object"])
        self.assertEqual(stored["repository"]["name"], "renamed")
        self.assertEqual(self.calls, [])

    def test_pooling_and_normalization_are_configured_and_invalidate(self):
        self.write()
        self.stage()
        self.run_embed()
        set_(self.root, "embedding.pooling", "mean")
        self.assertEqual(self.run_embed().generated, 1)
        set_(self.root, "embedding.normalize_embeddings", "false")
        self.assertEqual(self.run_embed().generated, 1)
        self.assertFalse(self.calls[-1][2])

    def test_full_provenance_and_effective_config_are_persisted(self):
        self.write()
        self.stage()
        self.run_embed()
        index = read_index(self.root)
        obj = read_object(self.root, "embeddings", self.reference()["vector"])
        self.assertEqual(obj["dimension"], 3)
        self.assertEqual(obj["provenance"]["model_revision"], REVISION)
        self.assertEqual(obj["embedding_config_hash"], index["embedding_config_hash"])
        self.assertNotIn("timestamp", obj)
        self.assertEqual(index["embedding_config"], index["embedding_config_hash"])

    def test_rename_and_readd_find_objects_without_cache_or_previous_reference(self):
        path = self.write()
        self.stage()
        self.run_embed()
        reference = self.reference()
        path.rename(path.with_name("renamed.txt"))
        self.calls.clear()
        self.assertEqual(self.run_embed().generated, 0)
        self.assertEqual(self.calls, [])
        self.assertEqual(read_index(self.root)["documents"]["data/renamed.txt"]["occurrences"][0]["embedding"], reference)

    def test_unpinned_revision_is_rejected_before_model_load(self):
        self.write()
        self.stage()
        for revision in ("null", "main", "v1.0", "deadbeef"):
            with self.subTest(revision=revision):
                set_(self.root, "model_revision", revision)
                with patch("embeddingvc.commands.embed.SentenceTransformerEncoder") as factory:
                    self.assert_preserved(lambda: embed(self.root), "40-character")
                    factory.assert_not_called()

    def test_model_loading_failure_preserves_index(self):
        self.write()
        self.stage()
        def fail(settings):
            raise EmbeddingError("model unavailable")
        self.assert_preserved(lambda: embed(self.root, encoder_factory=fail), "model unavailable")

    def test_invalid_vectors_never_publish_ready(self):
        self.write()
        self.stage()
        outputs = ([[1, 2]], [[1, 2, float("nan")]], [[1, float("inf"), 3]],
                   [[True, 2, 3]], [["1", 2, 3]], [], [[1, 2, 3], [1, 2, 3]])
        for output in outputs:
            with self.subTest(output=output):
                encoder = self.factory({"model": "fake", "revision": REVISION})
                encoder.encode = lambda *args, **kwargs: output
                self.assert_preserved(lambda: embed(self.root, encoder_factory=lambda _: encoder))
        self.assertEqual(self.objects(), [])

    def test_configured_dimension_mismatch_preserves_index(self):
        self.write()
        self.stage()
        set_(self.root, "embedding.dimension", "384")
        self.assert_preserved(self.run_embed, "dimension mismatch")
        self.assertEqual(self.calls, [])

    def test_corrupt_vector_chunk_and_config_are_never_reused_or_overwritten(self):
        self.write()
        self.stage()
        self.run_embed()
        index = read_index(self.root)
        occurrence = index["documents"]["data/a.txt"]["occurrences"][0]
        targets = [object_path(self.root, "embeddings", occurrence["embedding"]["vector"]),
                   object_path(self.root, "chunks", occurrence["chunk"]),
                   object_path(self.root, "configs", index["config"])]
        for path in targets:
            with self.subTest(kind=path.parent.name):
                original = path.read_bytes()
                path.write_text("{}", encoding="utf-8")
                self.assert_preserved(self.run_embed, "corrupt")
                self.assertEqual(path.read_text(), "{}")
                path.write_bytes(original)

    def test_missing_referenced_embedding_blocks_publication(self):
        self.write()
        self.stage()
        self.run_embed()
        object_path(self.root, "embeddings", self.reference()["vector"]).unlink()
        self.assert_preserved(self.run_embed, "Missing/corrupt")

    def test_failure_after_first_batch_keeps_objects_for_retry(self):
        self.write()
        self.write("b.txt", "distinct beta")
        self.stage()
        set_(self.root, "embedding.batch_size", "1")
        def fail_second():
            if len(self.calls) == 2:
                raise RuntimeError("model failed in second batch")
        self.callback = fail_second
        self.assert_preserved(self.run_embed, "model failed")
        self.assertEqual(len(self.objects()), 1)
        self.callback = None
        self.calls.clear()
        result = self.run_embed()
        self.assertEqual((result.generated, result.reused), (1, 1))
        self.assertEqual(len(self.calls), 1)

    def test_source_config_and_index_changes_during_encoding_preserve_index(self):
        path = self.write()
        self.stage()
        config_path = self.root / "embeddingvc.yaml"
        config_bytes = config_path.read_bytes()
        for mutate in (lambda: path.write_text("changed", encoding="utf-8"),
                       lambda: self.write("new.txt", "added during encoding"),
                       lambda: path.unlink(),
                       lambda: config_path.write_bytes(config_bytes + b"\n# edited\n")):
            with self.subTest(mutation=mutate):
                # Fresh revision forces an encoder call despite retry objects.
                set_(self.root, "model_revision", hash_text(str(mutate))[:40])
                self.callback = mutate
                self.assert_preserved(self.run_embed, "changed during embedding")
                self.write()
                (self.root / "data/new.txt").unlink(missing_ok=True)
                config_path.write_bytes(config_bytes)

    def test_external_index_edit_is_not_overwritten(self):
        self.write()
        self.stage()
        self.callback = lambda: index_path(self.root).write_bytes(b'{"version":1,"documents":{}}\n')
        with self.assertRaisesRegex(EmbedError, "changed during embedding"):
            self.run_embed()
        self.assertEqual(read_index(self.root)["documents"], {})

    def test_parse_failure_is_atomic(self):
        self.write()
        self.stage()
        self.write("invalid.txt").write_bytes(b"\xff")
        self.assert_preserved(self.run_embed, "UTF-8")

    def test_extraction_uses_the_same_bytes_as_the_document_hash(self):
        from embeddingvc.document_loader import load as real_load
        path = self.write()
        self.stage()
        def transient_edit(source, settings, **kwargs):
            original = source.read_bytes()
            source.write_text("transient different contents", encoding="utf-8")
            try:
                return real_load(source, settings, **kwargs)
            finally:
                source.write_bytes(original)
        with patch("embeddingvc.commands.add.load", side_effect=transient_edit):
            self.run_embed()
        self.assertEqual(self.calls[0][0], ["alpha beta"])

    def test_index_publication_failure_preserves_previous_index(self):
        self.write()
        self.stage()
        with patch("embeddingvc.objects.os.replace", side_effect=OSError("disk failure")):
            self.assert_preserved(self.run_embed, "disk failure")
        self.calls.clear()
        self.assertEqual(self.run_embed().generated, 0)
        self.assertEqual(self.calls, [])

    def test_shared_lock_blocks_embed_add_and_config(self):
        self.write()
        self.stage()
        with repository_lock(self.root):
            with self.assertRaisesRegex(EmbedError, "locked"):
                self.run_embed()
            with self.assertRaisesRegex(AddError, "locked"):
                self.stage()
            with self.assertRaisesRegex(ConfigurationError, "locked"):
                set_(self.root, "chunk_size", "200")
            self.assertTrue((self.root / ".embeddingvc/lock").exists())

    def test_no_history_or_database_writes_and_status_reports_ready(self):
        self.write()
        self.stage()
        paths = [self.root / ".embeddingvc/HEAD", self.root / ".embeddingvc/refs/heads/main"]
        originals = [path.read_bytes() for path in paths]
        self.run_embed()
        self.assertEqual([path.read_bytes() for path in paths], originals)
        self.assertEqual(list((self.root / ".embeddingvc/commits").iterdir()), [])
        self.assertEqual(list((self.root / "output/chroma").iterdir()), [])
        self.assertIn("Embeddings required: 0", status(self.root))
        self.objects()[0].write_text("{}", encoding="utf-8")
        self.assertIn("Embeddings required: 1", status(self.root))

    def test_add_after_embed_preserves_vectors_and_invalidates_changed_readiness(self):
        path = self.write()
        self.stage()
        self.run_embed()
        reference = self.reference()
        self.stage()
        self.assertEqual(self.reference(), reference)
        self.assertEqual(read_index(self.root)["generation"]["status"], "ready")
        path.write_text("updated", encoding="utf-8")
        self.stage()
        self.assertNotIn("generation", read_index(self.root))
        self.assertEqual(self.run_embed().generated, 1)

    def test_cli_registration_help_and_errors(self):
        for args, expected in ((["embed", "--help"], 0), (["embed", "extra"], 2), (["embed"], 1)):
            result = subprocess.run([sys.executable, "-m", "embeddingvc", *args], cwd=self.root,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, expected, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
        self.write()
        self.stage()
        with patch("embeddingvc.cli.embed", side_effect=lambda: self.run_embed()):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                app(["embed"])
        self.assertIn("chunk occurrences ready", output.getvalue())


@unittest.skipUnless(os.environ.get("EMBEDDINGVC_REAL_MODEL_REVISION"), "Real-model integration is opt-in")
class RealModelIntegrationTests(unittest.TestCase):
    def test_pinned_model_round_trip_and_no_change_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = initialize(Path(directory) / "repo").resolve()
            set_(root, "model_revision", os.environ["EMBEDDINGVC_REAL_MODEL_REVISION"])
            set_(root, "model", os.environ.get("EMBEDDINGVC_REAL_MODEL", "BAAI/bge-small-en-v1.5"))
            (root / "data/a.txt").write_text("A small integration document.", encoding="utf-8")
            add([str(root / "data")], directory=root)
            self.assertEqual(embed(root).generated, 1)
            with patch.object(SentenceTransformerEncoder, "encode", side_effect=AssertionError("unexpected encoding")):
                self.assertEqual(embed(root).generated, 0)
            self.assertIn("Embeddings required: 0", status(root))
