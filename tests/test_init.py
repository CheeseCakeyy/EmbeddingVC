import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embeddingvc.repository import InitializationError, initialize


class InitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "student notes"

    def test_layout_and_unborn_branch(self):
        initialize(self.root)
        for name in ("data", "output/chroma", ".embeddingvc/objects", ".embeddingvc/commits", ".embeddingvc/cache"):
            self.assertTrue((self.root / name).is_dir())
        self.assertEqual((self.root / ".embeddingvc/HEAD").read_text(), "ref: refs/heads/main\n")
        self.assertEqual((self.root / ".embeddingvc/refs/heads/main").read_bytes(), b"")
        self.assertEqual(json.loads((self.root / ".embeddingvc/index.json").read_text()), {"version": 1, "documents": {}})
        self.assertEqual(list((self.root / ".embeddingvc/commits").iterdir()), [])
        self.assertIn('name: "student notes"', (self.root / "embeddingvc.yaml").read_text())

    def test_reinitialization_even_with_force_preserves_history(self):
        initialize(self.root)
        before = (self.root / ".embeddingvc/HEAD").read_bytes()
        for force in (False, True):
            with self.assertRaises(InitializationError):
                initialize(self.root, force=force)
        self.assertEqual((self.root / ".embeddingvc/HEAD").read_bytes(), before)

    def test_existing_files_and_force(self):
        self.root.mkdir()
        (self.root / "README.md").write_text("User README")
        (self.root / "data").mkdir()
        source = self.root / "data/notes.txt"
        source.write_text("Keep me")
        with self.assertRaises(InitializationError):
            initialize(self.root)
        self.assertFalse((self.root / "output").exists())
        initialize(self.root, force=True)
        self.assertEqual(source.read_text(), "Keep me")

    def test_ignore_preservation(self):
        self.root.mkdir()
        ignore = self.root / ".gitignore"
        ignore.write_bytes(b"custom/\r\n/output/")
        initialize(self.root)
        self.assertTrue(ignore.read_bytes().startswith(b"custom/\r\n/output/\n"))
        self.assertEqual(ignore.read_text().count("/output/"), 1)

    def test_conflicting_directory(self):
        self.root.mkdir()
        (self.root / "output").write_text("keep")
        with self.assertRaises(InitializationError):
            initialize(self.root)
        self.assertEqual(list(self.root.iterdir()), [self.root / "output"])

    def test_file_target(self):
        self.root.write_text("keep")
        with self.assertRaises(InitializationError):
            initialize(self.root)

    def fail_head_open(self, path, *args, **kwargs):
        if path.name == "HEAD":
            raise PermissionError("injected failure")
        return self.original_open(path, *args, **kwargs)

    def test_rollback_new_nested_target(self):
        self.original_open = Path.open
        with patch.object(Path, "open", lambda path, *a, **kw: self.fail_head_open(path, *a, **kw)):
            with self.assertRaises(InitializationError):
                initialize(self.root / "nested")
        self.assertFalse(self.root.exists())

    def test_rollback_restores_existing_files(self):
        self.root.mkdir()
        readme = self.root / "README.md"
        readme.write_bytes(b"original\r\n")
        self.original_open = Path.open
        with patch.object(Path, "open", lambda path, *a, **kw: self.fail_head_open(path, *a, **kw)):
            with self.assertRaises(InitializationError):
                initialize(self.root, force=True)
        self.assertEqual(readme.read_bytes(), b"original\r\n")
        self.assertEqual(list(self.root.iterdir()), [readme])

    def test_cli_current_directory_and_error(self):
        result = subprocess.run([sys.executable, "-m", "embeddingvc", "init"], cwd=self.base, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Next steps", result.stdout)
        result = subprocess.run([sys.executable, "-m", "embeddingvc", "init"], cwd=self.base, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
