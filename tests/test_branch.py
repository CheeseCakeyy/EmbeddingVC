import tempfile
import unittest
from pathlib import Path

from embeddingvc.commands.branch import BranchError, branch
from embeddingvc.repository import initialize


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        initialize(self.root)

    def add_commit(self, commit="abcdef1234567890"):
        commits = self.root / ".embeddingvc" / "commits"
        (commits / commit).write_text("commit\n", encoding="utf-8")
        (self.root / ".embeddingvc/refs/heads/main").write_text(commit + "\n", encoding="utf-8")
        return commit

    def test_list_unborn_and_create_from_head(self):
        self.assertIn("* main", branch(self.root))
        self.assertIn("(unborn)", branch(self.root))
        commit = self.add_commit()
        self.assertIn("Created branch experiment at abcdef1", branch(self.root, "experiment"))
        self.assertEqual((self.root / ".embeddingvc/refs/heads/experiment").read_text().strip(), commit)
        self.assertIn("  experiment", branch(self.root))

    def test_explicit_revision_and_invalid_names_do_not_write(self):
        self.add_commit("1111111111111111")
        historical = self.root / ".embeddingvc/commits/2222222222222222"
        historical.write_text("commit\n", encoding="utf-8")
        branch(self.root, "historical", "2222222")
        heads = self.root / ".embeddingvc/refs/heads"
        before = sorted(path.name for path in heads.iterdir())
        for name in ("HEAD", "../escape", "deadbeef", "bad.lock"):
            with self.assertRaises(BranchError):
                branch(self.root, name)
        self.assertEqual(sorted(path.name for path in heads.iterdir()), before)

    def test_duplicate_is_case_insensitive_on_windows(self):
        self.add_commit()
        branch(self.root, "Experiment")
        if __import__("os").name == "nt":
            with self.assertRaises(BranchError):
                branch(self.root, "experiment")


if __name__ == "__main__":
    unittest.main()
