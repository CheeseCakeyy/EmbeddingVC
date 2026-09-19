import tempfile
import unittest
from pathlib import Path

from embeddingvc.config import parse
from embeddingvc.document_loader import DocumentError, extractor_versions, load

CONFIG = """
documents:
  source_directory: ./data
  supported_types: [txt, md, pdf]
preprocessing:
  unicode_normalization: NFKC
  remove_extra_whitespace: true
  lowercase: false
chunking:
  strategy: recursive-character
  chunk_size: 500
  chunk_overlap: 50
"""


def make_pdf(path, pages, *, password=None, image_only=False):
    import pymupdf

    document = pymupdf.open()
    for body in pages:
        page = document.new_page()
        if image_only:
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8))
            pixmap.clear_with(128)
            page.insert_image(pymupdf.Rect(10, 10, 60, 60), pixmap=pixmap)
        elif body:
            page.insert_text((72, 72), body)
    if password:
        document.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password)
    else:
        document.save(path)
    document.close()


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.config = parse(CONFIG)

    def write(self, name, text, encoding="utf-8"):
        path = self.base / name
        path.write_bytes(text.encode(encoding) if isinstance(text, str) else text)
        return path

    def test_plain_text(self):
        document = load(self.write("notes.txt", "Hello world"), self.config)
        self.assertEqual(document.text, "Hello world")
        self.assertEqual(document.extractor, "text")
        self.assertEqual(document.pages_for(0, 11), [])

    def test_markdown(self):
        document = load(self.write("notes.md", "# Title\n\nBody text"), self.config)
        self.assertEqual(document.text, "# Title\n\nBody text")

    def test_empty_file_loads_as_empty(self):
        document = load(self.write("empty.txt", ""), self.config)
        self.assertEqual(document.text, "")
        self.assertEqual(document.spans, ())

    def test_normalization_applied(self):
        document = load(self.write("messy.txt", "Café   spaced\r\n\r\n\r\nlines   here  "), self.config)
        self.assertEqual(document.text, "Café spaced\n\nlines here")

    def test_lowercase_setting_respected(self):
        config = parse(CONFIG.replace("lowercase: false", "lowercase: true"))
        document = load(self.write("caps.txt", "MiXeD Case"), config)
        self.assertEqual(document.text, "mixed case")

    def test_invalid_utf8_is_reported(self):
        path = self.base / "bad.txt"
        path.write_bytes(b"valid \xff\xfe invalid")
        with self.assertRaises(DocumentError) as caught:
            load(path, self.config)
        self.assertIn("UTF-8", str(caught.exception))

    def test_unsupported_extension_is_reported(self):
        with self.assertRaises(DocumentError):
            load(self.write("sheet.docx", "x"), self.config)

    def test_pdf_pages_and_provenance(self):
        path = self.base / "book.pdf"
        make_pdf(path, ["First page body", "Second page body"])
        document = load(path, self.config)
        self.assertIn("First page body", document.text)
        self.assertIn("Second page body", document.text)
        self.assertEqual(document.extractor, "pdf")
        self.assertEqual(document.pages_for(0, 5), [1])
        self.assertEqual(document.pages_for(0, len(document.text)), [1, 2])
        # Page text must not run together across the boundary.
        self.assertIn("\n\n", document.text)

    def test_pdf_with_no_pages_is_empty_not_an_error(self):
        path = self.base / "blank.pdf"
        make_pdf(path, [""])
        document = load(path, self.config)
        self.assertEqual(document.text, "")

    def test_encrypted_pdf_is_reported(self):
        path = self.base / "locked.pdf"
        make_pdf(path, ["secret"], password="hunter2")
        with self.assertRaises(DocumentError) as caught:
            load(path, self.config)
        self.assertIn("encrypted", str(caught.exception))

    def test_image_only_pdf_is_reported_as_unextractable(self):
        path = self.base / "scanned.pdf"
        make_pdf(path, ["ignored"], image_only=True)
        with self.assertRaises(DocumentError) as caught:
            load(path, self.config)
        self.assertIn("OCR", str(caught.exception))

    def test_corrupt_pdf_is_reported(self):
        path = self.base / "broken.pdf"
        path.write_bytes(b"%PDF-1.7 this is not a real pdf")
        with self.assertRaises(DocumentError):
            load(path, self.config)

    def test_extractor_versions_track_only_reachable_loaders(self):
        self.assertEqual(extractor_versions((".txt", ".md")), {"text": "1"})
        versions = extractor_versions((".txt", ".pdf"))
        self.assertEqual(set(versions), {"text", "pdf"})
        self.assertTrue(versions["pdf"].startswith("pymupdf-"))

    def test_loading_is_deterministic(self):
        path = self.base / "book.pdf"
        make_pdf(path, ["Alpha beta", "Gamma delta"])
        self.assertEqual(load(path, self.config).text, load(path, self.config).text)


if __name__ == "__main__":
    unittest.main()
