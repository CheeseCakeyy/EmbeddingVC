import unittest

from embeddingvc.chunking import DEFAULT_SEPARATORS, split_text


class ChunkingTests(unittest.TestCase):
    def test_separator_order_prefers_paragraphs(self):
        text = "\n\n".join("word " * 10 for _ in range(4)).strip()
        chunks = split_text(text, chunk_size=60, chunk_overlap=0)
        # Each paragraph is under the limit, so none should be broken mid-line.
        for chunk in chunks:
            self.assertNotIn("\n", chunk.text.strip("\n"))

    def test_offsets_index_the_source(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        for chunk in split_text(text, chunk_size=20, chunk_overlap=5):
            self.assertEqual(text[chunk.start:chunk.end], chunk.text)

    def test_chunks_cover_text_in_reading_order(self):
        text = " ".join(f"token{index}" for index in range(200))
        chunks = split_text(text, chunk_size=100, chunk_overlap=20)
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[-1].end, len(text))
        for earlier, later in zip(chunks, chunks[1:]):
            self.assertLess(earlier.start, later.start)

    def test_overlap_repeats_trailing_context(self):
        text = " ".join(f"token{index}" for index in range(100))
        overlapped = split_text(text, chunk_size=100, chunk_overlap=40)
        self.assertTrue(any(later.start < earlier.end
                            for earlier, later in zip(overlapped, overlapped[1:])))
        separate = split_text(text, chunk_size=100, chunk_overlap=0)
        for earlier, later in zip(separate, separate[1:]):
            self.assertGreaterEqual(later.start, earlier.end)

    def test_respects_chunk_size(self):
        text = " ".join(f"token{index}" for index in range(300))
        for chunk in split_text(text, chunk_size=120, chunk_overlap=30):
            self.assertLessEqual(len(chunk.text), 120)

    def test_character_fallback_for_unbroken_text(self):
        text = "x" * 250
        chunks = split_text(text, chunk_size=100, chunk_overlap=0)
        self.assertEqual([len(chunk.text) for chunk in chunks], [100, 100, 50])
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_deterministic_across_runs(self):
        text = "Para one.\n\n" + "sentence " * 80 + "\n\nPara three."
        first = split_text(text, chunk_size=150, chunk_overlap=30)
        second = split_text(text, chunk_size=150, chunk_overlap=30)
        self.assertEqual([(c.text, c.start, c.end) for c in first],
                         [(c.text, c.start, c.end) for c in second])

    def test_empty_and_whitespace_text_yields_nothing(self):
        self.assertEqual(split_text("", 100, 10), [])
        self.assertEqual(split_text("   \n\n  ", 100, 10), [])

    def test_text_shorter_than_chunk_size_is_one_chunk(self):
        chunks = split_text("short", 100, 10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "short")

    def test_invalid_settings_rejected(self):
        for size, overlap in ((0, 0), (-1, 0), (100, -1), (100, 100), (100, 200)):
            with self.assertRaises(ValueError):
                split_text("text", size, overlap)

    def test_default_separator_order(self):
        self.assertEqual(DEFAULT_SEPARATORS, ("\n\n", "\n", " ", ""))

    def test_large_piece_still_starts_a_chunk(self):
        # A run longer than chunk_size must not stall the merge loop.
        text = "tiny " + "y" * 300 + " tiny"
        chunks = split_text(text, chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[-1].end, len(text))


if __name__ == "__main__":
    unittest.main()
