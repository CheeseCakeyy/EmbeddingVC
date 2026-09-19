import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embeddingvc.commands.add import AddError, add
from embeddingvc.objects import index_path, object_path, read_index
from embeddingvc.repository import initialize
from test_document_loader import make_pdf


class AddTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "notes"
        initialize(self.root)
        self.data = self.root / "data"

    def write(self, name, text):
        path = self.data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def index(self):
        return read_index(self.root)

    def add(self, *paths):
        return add([str(path) for path in paths], directory=self.root)


class StagingTests(AddTestCase):
    def test_stages_text_markdown_pdf_and_empty_files(self):
        self.write("notes.txt", "alpha beta gamma")
        self.write("chapter.md", "# Title\n\nSome body text here.")
        self.write("empty.txt", "")
        make_pdf(self.data / "book.pdf", ["Page one text", "Page two text"])

        result = self.add(self.data)
        documents = self.index()["documents"]
        self.assertEqual(sorted(documents), ["data/book.pdf", "data/chapter.md", "data/empty.txt", "data/notes.txt"])
        self.assertEqual(result.added, 4)
        self.assertEqual(documents["data/empty.txt"]["occurrences"], [])
        self.assertTrue(documents["data/book.pdf"]["occurrences"])
        self.assertEqual(documents["data/book.pdf"]["extractor"], "pdf")

    def test_pdf_occurrences_carry_page_numbers(self):
        make_pdf(self.data / "book.pdf", ["Alpha page one", "Beta page two"])
        self.add(self.data)
        pages = [occurrence["pages"] for occurrence in self.index()["documents"]["data/book.pdf"]["occurrences"]]
        self.assertTrue(all(page for page in pages))
        self.assertLessEqual(set(number for page in pages for number in page), {1, 2})

    def test_text_occurrences_have_no_pages(self):
        self.write("notes.txt", "alpha beta")
        self.add(self.data)
        for occurrence in self.index()["documents"]["data/notes.txt"]["occurrences"]:
            self.assertEqual(occurrence["pages"], [])

    def test_chunk_objects_written_and_readable(self):
        self.write("notes.txt", "alpha beta gamma delta")
        self.add(self.data)
        for occurrence in self.index()["documents"]["data/notes.txt"]["occurrences"]:
            stored = object_path(self.root, "chunks", occurrence["chunk"])
            self.assertTrue(stored.is_file())
            self.assertIn("text", json.loads(stored.read_text(encoding="utf-8")))

    def test_configuration_snapshot_persisted(self):
        self.write("notes.txt", "alpha")
        self.add(self.data)
        digest = self.index()["config"]
        self.assertIsNotNone(digest)
        snapshot = json.loads(object_path(self.root, "configs", digest).read_text(encoding="utf-8"))
        self.assertEqual(snapshot["chunking"]["chunk_size"], 500)
        self.assertIn("extractors", snapshot)

    def test_no_model_or_vector_store_writes(self):
        self.write("notes.txt", "alpha beta")
        make_pdf(self.data / "book.pdf", ["text"])
        self.add(self.data)
        self.assertEqual(list((self.root / "output/chroma").iterdir()), [])
        for occurrence in self.index()["documents"]["data/notes.txt"]["occurrences"]:
            self.assertIsNone(occurrence["embedding"])


class DeterminismTests(AddTestCase):
    def test_repeated_scans_are_identical(self):
        self.write("notes.txt", "sentence one. " * 60)
        self.write("chapter.md", "# Heading\n\n" + "body text " * 80)
        make_pdf(self.data / "book.pdf", ["Alpha " * 40, "Beta " * 40])

        self.add(self.data)
        first = index_path(self.root).read_bytes()
        result = self.add(self.data)
        self.assertEqual(index_path(self.root).read_bytes(), first)
        self.assertEqual((result.added, result.modified, result.deleted), (0, 0, 0))

    def test_occurrence_ids_and_order_are_stable(self):
        self.write("notes.txt", "alpha beta gamma " * 50)
        self.add(self.data)
        before = self.index()["documents"]["data/notes.txt"]["occurrences"]
        self.add(self.data)
        after = self.index()["documents"]["data/notes.txt"]["occurrences"]
        self.assertEqual([item["id"] for item in before], [item["id"] for item in after])
        self.assertEqual([item["ordinal"] for item in after], list(range(len(after))))

    def test_modified_file_is_detected(self):
        self.write("notes.txt", "original text")
        self.add(self.data)
        self.write("notes.txt", "replacement text entirely")
        result = self.add(self.data)
        self.assertEqual((result.added, result.modified, result.deleted), (0, 1, 0))


class DeduplicationTests(AddTestCase):
    def test_duplicate_files_share_one_chunk_object(self):
        body = "the same words repeated exactly"
        self.write("first.txt", body)
        self.write("second.txt", body)
        self.add(self.data)
        documents = self.index()["documents"]
        self.assertEqual(
            [item["chunk"] for item in documents["data/first.txt"]["occurrences"]],
            [item["chunk"] for item in documents["data/second.txt"]["occurrences"]],
        )
        self.assertNotEqual(
            [item["id"] for item in documents["data/first.txt"]["occurrences"]],
            [item["id"] for item in documents["data/second.txt"]["occurrences"]],
        )
        self.assertEqual(len(list(object_path(self.root, "chunks", "x").parent.iterdir())), 1)

    def test_repeated_text_within_a_document_keeps_every_occurrence(self):
        # Each paragraph must be long enough to become a chunk of its own,
        # otherwise the four of them merge into one and nothing is repeated.
        paragraph = ("word " * 96).strip()
        self.write("repeats.txt", "\n\n".join([paragraph] * 4))
        self.add(self.data)
        occurrences = self.index()["documents"]["data/repeats.txt"]["occurrences"]
        digests = [item["chunk"] for item in occurrences]
        self.assertGreater(len(occurrences), 1)
        self.assertEqual(len(set(digests)), 1, "repeated text should share one object")
        self.assertEqual(len(set(item["id"] for item in occurrences)), len(occurrences))


class ScopeTests(AddTestCase):
    def test_directory_rescan_detects_deletion(self):
        self.write("keep.txt", "keep this")
        removed = self.write("gone.txt", "remove this")
        self.add(self.data)
        removed.unlink()
        result = self.add(self.data)
        self.assertEqual(result.deleted, 1)
        self.assertEqual(sorted(self.index()["documents"]), ["data/keep.txt"])

    def test_adding_one_file_preserves_other_staged_files(self):
        self.write("first.txt", "first body")
        self.add(self.data)
        second = self.write("second.txt", "second body")
        result = self.add(second)
        self.assertEqual((result.added, result.deleted), (1, 0))
        self.assertEqual(sorted(self.index()["documents"]), ["data/first.txt", "data/second.txt"])

    def test_single_file_scope_does_not_delete_outside_itself(self):
        first = self.write("first.txt", "first body")
        self.write("second.txt", "second body")
        self.add(self.data)
        (self.data / "second.txt").unlink()
        result = self.add(first)
        self.assertEqual(result.deleted, 0)
        self.assertIn("data/second.txt", self.index()["documents"])

    def test_missing_tracked_file_stages_deletion(self):
        path = self.write("gone.txt", "temporary")
        self.add(self.data)
        path.unlink()
        result = self.add(path)
        self.assertEqual(result.deleted, 1)
        self.assertNotIn("data/gone.txt", self.index()["documents"])

    def test_unknown_missing_path_is_an_error(self):
        with self.assertRaises(AddError) as caught:
            self.add(self.data / "never-existed.txt")
        self.assertIn("does not exist", str(caught.exception))

    def test_nested_roots_do_not_duplicate_documents(self):
        nested = self.write("chapters/one.txt", "chapter one body")
        self.add(self.data, nested.parent, nested)
        documents = self.index()["documents"]
        self.assertEqual(sorted(documents), ["data/chapters/one.txt"])
        self.assertEqual(self.index()["tracked_roots"], ["data"])

    def test_tracked_roots_persist_and_prune(self):
        self.write("chapters/one.txt", "body")
        self.add(self.data / "chapters")
        self.assertEqual(self.index()["tracked_roots"], ["data/chapters"])
        self.add(self.data)
        self.assertEqual(self.index()["tracked_roots"], ["data"])

    def test_excluded_directories_are_not_scanned(self):
        (self.root / "reports").mkdir()
        (self.root / "reports/summary.md").write_text("generated", encoding="utf-8")
        (self.root / "output/note.txt").write_text("generated", encoding="utf-8")
        self.write("real.txt", "real body")
        self.add(self.root)
        # The generated README at the root is repository documentation, not a
        # document; one inside data/ is an ordinary file and is staged.
        self.write("notes/README.md", "a real document")
        self.add(self.root)
        self.assertEqual(sorted(self.index()["documents"]), ["data/notes/README.md", "data/real.txt"])

    def test_generated_root_file_named_directly_is_refused(self):
        with self.assertRaises(AddError) as caught:
            self.add(self.root / "README.md")
        self.assertIn("generated by EmbeddingVC", str(caught.exception))

    def test_managed_directory_cannot_be_added(self):
        with self.assertRaises(AddError):
            self.add(self.root / ".embeddingvc")

    def test_unsupported_files_are_skipped_and_reported(self):
        self.write("real.txt", "body")
        (self.data / "image.png").write_bytes(b"\x89PNG\r\n")
        result = self.add(self.data)
        self.assertEqual(result.skipped, ["data/image.png"])
        self.assertNotIn("data/image.png", self.index()["documents"])

    def test_unsupported_file_named_directly_is_an_error(self):
        path = self.data / "image.png"
        path.write_bytes(b"\x89PNG\r\n")
        with self.assertRaises(AddError):
            self.add(path)


class SafetyTests(AddTestCase):
    def test_path_outside_repository_is_rejected(self):
        outside = self.base / "outside.txt"
        outside.write_text("not ours", encoding="utf-8")
        with self.assertRaises(AddError) as caught:
            self.add(outside)
        self.assertIn("outside the repository", str(caught.exception))

    def test_parent_traversal_escape_is_rejected(self):
        outside = self.base / "outside.txt"
        outside.write_text("not ours", encoding="utf-8")
        with self.assertRaises(AddError):
            self.add(self.data / ".." / ".." / "outside.txt")

    @unittest.skipUnless(hasattr(os, "symlink"), "requires symlink support")
    def test_symlinked_file_is_rejected(self):
        target = self.base / "outside.txt"
        target.write_text("not ours", encoding="utf-8")
        link = self.data / "link.txt"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not permitted on this platform")
        with self.assertRaises(AddError) as caught:
            self.add(self.data)
        self.assertIn("Linked path", str(caught.exception))

    @unittest.skipUnless(hasattr(os, "symlink"), "requires symlink support")
    def test_symlinked_directory_is_rejected(self):
        target = self.base / "elsewhere"
        target.mkdir()
        (target / "note.txt").write_text("not ours", encoding="utf-8")
        link = self.data / "linked"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not permitted on this platform")
        with self.assertRaises(AddError):
            self.add(self.data)

    def test_parser_failure_preserves_prior_index(self):
        self.write("good.txt", "good body")
        self.add(self.data)
        before = index_path(self.root).read_bytes()
        (self.data / "broken.pdf").write_bytes(b"%PDF-1.7 not actually a pdf")
        with self.assertRaises(AddError):
            self.add(self.data)
        self.assertEqual(index_path(self.root).read_bytes(), before)

    def test_undecodable_file_preserves_prior_index(self):
        self.write("good.txt", "good body")
        self.add(self.data)
        before = index_path(self.root).read_bytes()
        (self.data / "bad.txt").write_bytes(b"\xff\xfe\x00broken")
        with self.assertRaises(AddError):
            self.add(self.data)
        self.assertEqual(index_path(self.root).read_bytes(), before)

    def test_failure_does_not_stage_deletions(self):
        keep = self.write("keep.txt", "keep this")
        self.add(self.data)
        keep.unlink()
        (self.data / "bad.txt").write_bytes(b"\xff\xfe\x00broken")
        with self.assertRaises(AddError):
            self.add(self.data)
        self.assertIn("data/keep.txt", self.index()["documents"])

    def test_outside_a_repository_is_an_error(self):
        with self.assertRaises(AddError) as caught:
            add([str(self.base)], directory=self.base)
        self.assertIn("Not an EmbeddingVC repository", str(caught.exception))

    def test_no_paths_is_an_error(self):
        with self.assertRaises(AddError):
            add([], directory=self.root)


class ConfigurationChangeTests(AddTestCase):
    def rewrite_config(self, old, new):
        path = self.root / "embeddingvc.yaml"
        path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")

    def test_chunk_size_change_restages(self):
        self.write("notes.txt", "word " * 400)
        self.add(self.data)
        before = self.index()
        self.rewrite_config("chunk_size: 500", "chunk_size: 200")
        self.add(self.data)
        after = self.index()
        self.assertNotEqual(before["config"], after["config"])
        self.assertNotEqual(
            [item["chunk"] for item in before["documents"]["data/notes.txt"]["occurrences"]],
            [item["chunk"] for item in after["documents"]["data/notes.txt"]["occurrences"]],
        )

    def test_invalid_configuration_is_reported(self):
        self.write("notes.txt", "body")
        self.rewrite_config("chunk_overlap: 50", "chunk_overlap: 900")
        with self.assertRaises(AddError) as caught:
            self.add(self.data)
        self.assertIn("chunk_overlap", str(caught.exception))

    def test_compatible_embedding_reference_is_carried_forward(self):
        self.write("notes.txt", "stable body text")
        self.add(self.data)
        index = self.index()
        entry = index["documents"]["data/notes.txt"]
        from embeddingvc.config import load as load_config
        fingerprint = load_config(self.root).embedding_fingerprint()
        entry["occurrences"][0]["embedding"] = {"vector": "vec-1", "fingerprint": fingerprint}
        index_path(self.root).write_text(json.dumps(index), encoding="utf-8")

        # An unrelated file changes, forcing this document through the reuse path.
        self.write("other.txt", "another body")
        self.add(self.data)
        carried = self.index()["documents"]["data/notes.txt"]["occurrences"][0]["embedding"]
        self.assertEqual(carried, {"vector": "vec-1", "fingerprint": fingerprint})

    def test_incompatible_embedding_reference_is_dropped(self):
        self.write("notes.txt", "stable body text")
        self.add(self.data)
        index = self.index()
        index["documents"]["data/notes.txt"]["occurrences"][0]["embedding"] = {
            "vector": "vec-1", "fingerprint": "a-different-model",
        }
        index_path(self.root).write_text(json.dumps(index), encoding="utf-8")
        self.rewrite_config("chunk_size: 500", "chunk_size: 300")
        self.add(self.data)
        self.assertIsNone(self.index()["documents"]["data/notes.txt"]["occurrences"][0]["embedding"])


class OutputTests(AddTestCase):
    def test_report_format(self):
        self.write("one.txt", "first body")
        self.write("two.txt", "second body")
        report = self.add(self.data).render()
        lines = report.splitlines()
        self.assertEqual(lines[0], "Staged data/: 2 documents, 2 chunk occurrences")
        self.assertEqual(lines[1], "Documents: 2 added, 0 modified, 0 deleted")
        self.assertEqual(lines[-1], "Next: embeddingvc status")

    def test_singular_wording(self):
        self.write("one.txt", "only body")
        report = self.add(self.data).render()
        self.assertIn("1 document, 1 chunk occurrence", report)

    def test_skip_notice_included(self):
        self.write("one.txt", "body")
        (self.data / "image.png").write_bytes(b"\x89PNG\r\n")
        self.assertIn("Skipped 1 unsupported file: data/image.png", self.add(self.data).render())


class CommandLineTests(AddTestCase):
    def run_cli(self, *arguments, cwd=None):
        environment = dict(os.environ)
        source = str(Path(__file__).resolve().parents[1] / "src")
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, "-m", "embeddingvc", *arguments],
            cwd=str(cwd or self.root), capture_output=True, text=True, env=environment,
        )

    def test_cli_stages_relative_path(self):
        self.write("notes.txt", "body text")
        result = self.run_cli("add", "./data")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Staged data/: 1 document", result.stdout)
        self.assertIn("Next: embeddingvc status", result.stdout)

    def test_cli_accepts_several_paths(self):
        first = self.write("one.txt", "first")
        second = self.write("two.md", "second")
        result = self.run_cli("add", str(first), str(second))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("Staged "), 2)

    def test_cli_reports_errors_without_traceback(self):
        result = self.run_cli("add", "./nope.txt")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("Error:", result.stderr)

    def test_cli_requires_a_path(self):
        self.assertEqual(self.run_cli("add").returncode, 2)

    def test_cli_outside_repository_fails_cleanly(self):
        result = self.run_cli("add", ".", cwd=self.base)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
