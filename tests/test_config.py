import json
import tempfile
import unittest
from pathlib import Path

from embeddingvc.commands.config import get, set_, show
from embeddingvc.config import ConfigurationError, load
from embeddingvc.repository import initialize


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        initialize(self.root)

    def test_show_get_and_aliases(self):
        self.assertIn("chunk_size: 500", show(self.root))
        self.assertEqual(get(self.root, "model"), "BAAI/bge-small-en-v1.5")
        self.assertEqual(get(self.root, "embedding.model"), get(self.root, "model"))
        set_(self.root, "chunk_size", "700")
        self.assertEqual(get(self.root, "chunking.chunk_size"), "700")

    def test_invalid_update_leaves_yaml_unchanged(self):
        path = self.root / "embeddingvc.yaml"
        before = path.read_bytes()
        with self.assertRaises(ConfigurationError):
            set_(self.root, "chunk_overlap", "700")
        self.assertEqual(path.read_bytes(), before)

    def test_effective_hash_ignores_batch_name_and_database_location(self):
        initial = load(self.root).effective_hash
        set_(self.root, "embedding.batch_size", "64")
        self.assertEqual(load(self.root).effective_hash, initial)
        set_(self.root, "repository.name", "renamed")
        self.assertEqual(load(self.root).effective_hash, initial)
        set_(self.root, "vector_store.persist_directory", "output/other")
        self.assertEqual(load(self.root).effective_hash, initial)
        set_(self.root, "chunk_size", "700")
        self.assertNotEqual(load(self.root).effective_hash, initial)

    def test_paths_and_provider_are_validated(self):
        with self.assertRaises(ConfigurationError):
            set_(self.root, "documents.source_directory", "../outside")
        with self.assertRaises(ConfigurationError):
            set_(self.root, "vector_store.provider", "other")

    def test_identity_is_not_pipeline_configuration(self):
        identity = json.loads((self.root / ".embeddingvc/config.json").read_text())
        self.assertEqual(identity, {"repository": {"name": "repo"}})
        self.assertNotIn("chunking", identity)


if __name__ == "__main__":
    unittest.main()
