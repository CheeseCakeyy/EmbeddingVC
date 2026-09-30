import unittest
from embeddingvc.change_detector import compare


def doc(*chunks, content="raw"):
    return {"content_hash": content, "occurrences": [{"chunk": c} for c in chunks]}


class ComparisonTests(unittest.TestCase):
    def test_all_classifications(self):
        old = {"a": doc("keep", "stale", "replace", "remove"), "deleted": doc("gone")}
        new = {"a": doc("keep", "stale", "new", content="edited"), "added": doc("extra")}
        result = compare(old, new, {"stale"})
        self.assertEqual(result["chunks"], dict(added=1, modified=1, deleted=2, unchanged=1, stale=1))
        self.assertEqual(result["documents"], dict(added=1, modified=1, deleted=1, unchanged=0))

    def test_duplicates_and_shifted_boundaries_match_content_first(self):
        old = {"a": doc("same", "same", "tail")}
        new = {"a": doc("prefix", "same", "tail", "same")}
        self.assertEqual(compare(old, new)["chunks"], dict(added=1, modified=0, deleted=0, unchanged=3, stale=0))
