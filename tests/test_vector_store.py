import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

from embeddingvc.vector_store.chromadb_adapter import ChromaAdapter
from embeddingvc.objects import RepositoryError


class Collection:
    def __init__(self):
        self.rows = []
        self.corruption = None

    def add(self, ids, embeddings, documents, metadatas):
        self.rows.extend(zip(ids, embeddings, documents, metadatas))

    def count(self):
        return len(self.rows)

    def get(self, limit, offset, include):
        rows = copy.deepcopy(self.rows[offset:offset + limit])
        result = {key: [row[i] for row in rows] for i, key in
                  enumerate(("ids", "embeddings", "documents", "metadatas"))}
        if self.corruption:
            self.corruption(result)
        return result


class Client:
    def __init__(self):
        self.collections = {}
        self.corruption = None

    def create_collection(self, name, embedding_function, metadata):
        assert embedding_function is None
        collection = Collection()
        collection.corruption = self.corruption
        self.collections[name] = collection
        return collection

    def get_max_batch_size(self):
        return 1  # Force multi-batch insertion and verification.


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.client = Client()
        self.settings = {"persist_directory": "output/chroma", "collection": "test"}
        self.adapter = ChromaAdapter(self.root, self.settings, client=self.client)
        self.rows = [{"id": str(i), "text": "duplicate", "vector": [1., 0., 0.],
                      "metadata": {"source": f"data/{i}.txt"}} for i in range(2)]

    def test_explicit_vectors_replacement_and_empty_collection(self):
        first = self.adapter.replace("a" * 64, self.rows, 3)
        second = self.adapter.replace("b" * 64, self.rows[1:], 3)
        empty = self.adapter.replace("c" * 64, [], 3)
        self.assertEqual(self.client.collections[first].count(), 2)
        self.assertEqual(self.client.collections[second].rows[0][0], "1")
        self.assertEqual(self.client.collections[empty].count(), 0)

    def test_membership_dimension_text_metadata_and_values_verified(self):
        for key, value in (("ids", "wrong"), ("embeddings", [1., 0.]),
                           ("embeddings", [0., 1., 0.]), ("embeddings", [float("nan"), 0., 0.]),
                           ("documents", "wrong"), ("metadatas", {"source": "wrong"})):
            with self.subTest(key=key, value=value):
                self.client.corruption = lambda page: page[key].__setitem__(0, value)
                with self.assertRaises(RepositoryError):
                    self.adapter.replace("a" * 64, self.rows, 3)

    def test_invalid_input_never_builds_collection(self):
        for rows in ([self.rows[0], self.rows[0]], [{**self.rows[0], "vector": [1.]}]):
            with self.assertRaises(RepositoryError):
                self.adapter.replace("a" * 64, rows, 3)
        self.assertEqual(self.client.collections, {})

    def test_unsafe_paths_rejected(self):
        for path in ("../outside", ".embeddingvc/chroma", "."):
            with self.assertRaises(RepositoryError):
                ChromaAdapter(self.root, {**self.settings, "persist_directory": path}, client=self.client)

    @unittest.skipUnless(importlib.util.find_spec("chromadb"), "Install .[vector-store] for real Chroma integration")
    def test_real_chroma_replacement_roundtrip_and_reopen(self):
        # Run the real client in a subprocess: Windows keeps SQLite files open
        # for the client's lifetime, so process exit precedes temp cleanup.
        import subprocess
        import sys
        script = '''
import sys
from pathlib import Path
from embeddingvc.vector_store.chromadb_adapter import ChromaAdapter
root = Path(sys.argv[1])
settings = {"persist_directory": "output/chroma", "collection": "integration"}
a = ChromaAdapter(root, settings)
rows = [{"id": str(i), "text": "duplicate", "vector": [1., 0., 0.], "metadata": {"source": str(i)}} for i in range(2)]
first = a.replace("a" * 64, rows, 3)
second = a.replace("b" * 64, rows[1:], 3)
empty = a.replace("c" * 64, [], 3)
b = ChromaAdapter(root, settings)
assert b.client.get_collection(first, embedding_function=None).count() == 2
assert b.client.get_collection(second, embedding_function=None).get()["ids"] == ["1"]
assert b.client.get_collection(empty, embedding_function=None).count() == 0
'''
        result = subprocess.run([sys.executable, "-c", script, str(self.root)], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
