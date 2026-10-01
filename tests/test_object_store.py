import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embeddingvc.hashing import canonical_json, hash_payload
from embeddingvc.object_store import find_embeddings, read_object, write_object
from embeddingvc.objects import RepositoryError, object_path
from embeddingvc.repository import initialize


class ObjectStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = initialize(Path(self.temp.name) / "repo").resolve()
        self.payload = {"schema": 1, "chunk_hash": "a" * 64,
                        "embedding_config_hash": "b" * 64, "dimension": 2,
                        "vector": [0.2, 0.4], "provenance": {"provider": "fake"}}

    def test_round_trip_and_repeated_write_keeps_original_bytes(self):
        digest = write_object(self.root, "embeddings", self.payload)
        path = object_path(self.root, "embeddings", digest)
        self.assertEqual(read_object(self.root, "embeddings", digest), self.payload)
        before = path.stat().st_mtime_ns
        self.assertEqual(write_object(self.root, "embeddings", self.payload), digest)
        self.assertEqual(path.stat().st_mtime_ns, before)

    def test_hash_mismatch_and_invalid_json_cannot_be_overwritten(self):
        digest = write_object(self.root, "embeddings", self.payload)
        path = object_path(self.root, "embeddings", digest)
        for corrupt in ("{", "{}", "[]", '{"vector":[NaN]}'):
            with self.subTest(corrupt=corrupt):
                path.write_text(corrupt, encoding="utf-8")
                with self.assertRaises(RepositoryError):
                    read_object(self.root, "embeddings", digest)
                with self.assertRaises(RepositoryError):
                    write_object(self.root, "embeddings", self.payload)
                self.assertEqual(path.read_text(), corrupt)

    def test_nonfinite_json_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                canonical_json({"value": value})

    def test_invalid_vector_payloads_fail_without_writing(self):
        for changed in ({"vector": [1]}, {"vector": [False, 1]}, {"vector": [float("nan"), 1]},
                        {"dimension": 0}, {"dimension": True}, {"provenance": {}}, {"chunk_hash": "../escape"}):
            with self.subTest(changed=changed):
                with self.assertRaises(RepositoryError):
                    write_object(self.root, "embeddings", {**self.payload, **changed})
        self.assertFalse((self.root / ".embeddingvc/objects/embeddings").exists())

    def test_semantically_invalid_object_with_valid_hash_is_rejected(self):
        invalid = {**self.payload, "vector": [1]}
        digest = hash_payload(invalid)
        path = object_path(self.root, "embeddings", digest)
        path.parent.mkdir()
        path.write_text(json.dumps(invalid), encoding="utf-8")
        with self.assertRaises(RepositoryError):
            read_object(self.root, "embeddings", digest)

    def test_lookup_uses_pair_and_checks_dimension_provenance(self):
        digest = write_object(self.root, "embeddings", self.payload)
        found = find_embeddings(self.root, "b" * 64, 2, self.payload["provenance"])
        self.assertEqual(found, {"a" * 64: digest})
        self.assertEqual(find_embeddings(self.root, "c" * 64, 2, self.payload["provenance"]), {})
        with self.assertRaises(RepositoryError):
            find_embeddings(self.root, "b" * 64, 3, self.payload["provenance"])
        with self.assertRaises(RepositoryError):
            find_embeddings(self.root, "b" * 64, 2, {"provider": "different"})

    def test_install_failure_leaves_no_object_or_temp_file(self):
        with patch("embeddingvc.object_store.os.link", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                write_object(self.root, "embeddings", self.payload)
        self.assertEqual(list((self.root / ".embeddingvc/objects/embeddings").iterdir()), [])

    def test_path_traversal_and_unknown_kind_rejected(self):
        for kind, digest in (("../../outside", "a" * 64), ("embeddings", "../outside"), ("embeddings", "A" * 64)):
            with self.assertRaises(RepositoryError):
                read_object(self.root, kind, digest)

    def test_linked_object_directory_is_rejected(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        path = self.root / ".embeddingvc/objects/embeddings"
        try:
            path.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Symbolic links unavailable")
        with self.assertRaisesRegex(RepositoryError, "Linked"):
            write_object(self.root, "embeddings", self.payload)
        self.assertEqual(list(outside.iterdir()), [])
