import types
import unittest
from unittest.mock import MagicMock, patch

from embeddingvc.embedding_engine import EmbeddingError, SentenceTransformerEncoder


class FakePooling:
    def __init__(self, width=8, pooling_mode="mean", include_prompt=True):
        self.width = width
        self.mode = pooling_mode
        self.include_prompt = include_prompt

    def get_config_dict(self):
        return {"word_embedding_dimension": self.width, "pooling_mode": self.mode}


class EncoderAdapterTests(unittest.TestCase):
    def setUp(self):
        self.settings = {"model": "fake/model", "revision": "a" * 40,
                         "normalize_embeddings": True, "pooling": "model-default"}
        config = MagicMock()
        config._commit_hash = self.settings["revision"]
        config.to_dict.return_value = {"hidden_size": 8}
        transformer = MagicMock()
        transformer.auto_model.config = config
        transformer.get_config_dict.return_value = {"max_seq_length": 128}
        tokenizer = MagicMock()
        tokenizer.backend_tokenizer.to_str.return_value = '{"tokenizer":"fake"}'
        tokenizer.model_max_length = 128
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        self.model = MagicMock()
        self.model._modules = {"0": transformer, "1": FakePooling()}
        self.model._first_module.return_value = transformer
        self.model.get_sentence_embedding_dimension.return_value = 8
        self.model.tokenizer = tokenizer
        self.model.max_seq_length = 128
        self.model.prompts = {}
        self.model.default_prompt_name = None
        self.constructor = MagicMock(return_value=self.model)
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = self.constructor
        module.models = types.SimpleNamespace(Pooling=FakePooling)
        self.modules_patch = patch.dict("sys.modules", {"sentence_transformers": module})
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        self.version_patch = patch("embeddingvc.embedding_engine.version", return_value="test-version")
        self.version_patch.start()
        self.addCleanup(self.version_patch.stop)

    def test_pinned_loading_and_provenance_do_not_encode(self):
        encoder = SentenceTransformerEncoder(self.settings)
        self.constructor.assert_called_once_with("fake/model", revision="a" * 40,
                                                 trust_remote_code=False, device="cpu")
        self.model.encode.assert_not_called()
        self.model.float.assert_called_once_with()
        self.assertEqual(encoder.dimension, 8)
        self.assertEqual(encoder.provenance["tokenizer"]["revision"], "a" * 40)
        self.assertEqual(encoder.provenance["libraries"]["torch"], "test-version")

    def test_encode_passes_normalization_batch_and_float32(self):
        self.model.encode.return_value.tolist.return_value = [[1.0] * 8]
        encoder = SentenceTransformerEncoder(self.settings)
        self.assertEqual(encoder.encode(["text"], batch_size=4, normalize_embeddings=True), [[1.0] * 8])
        self.model.encode.assert_called_once_with(["text"], batch_size=4, show_progress_bar=False,
                                                 convert_to_numpy=True, normalize_embeddings=True,
                                                 precision="float32")

    def test_pooling_override_is_applied_to_model(self):
        encoder = SentenceTransformerEncoder({**self.settings, "pooling": "max"})
        self.assertEqual(encoder.model._modules["1"].mode, "max")
        self.assertEqual(encoder.provenance["modules"][1]["config"]["pooling_mode"], "max")

    def test_dimension_and_loaded_revision_are_verified(self):
        with self.assertRaisesRegex(EmbeddingError, "dimension mismatch"):
            SentenceTransformerEncoder({**self.settings, "dimension": 384})
        self.model._first_module().auto_model.config._commit_hash = "b" * 40
        with self.assertRaisesRegex(EmbeddingError, "revision"):
            SentenceTransformerEncoder(self.settings)

    def test_invalid_dimension_and_unsupported_model_pooling_fail(self):
        self.model.get_sentence_embedding_dimension.return_value = None
        with self.assertRaisesRegex(EmbeddingError, "positive embedding dimension"):
            SentenceTransformerEncoder(self.settings)
        self.model.get_sentence_embedding_dimension.return_value = 8
        self.model._modules = {}
        with self.assertRaisesRegex(EmbeddingError, "exactly one Pooling"):
            SentenceTransformerEncoder({**self.settings, "pooling": "mean"})

    def test_model_and_encoding_failures_are_actionable(self):
        self.constructor.side_effect = RuntimeError("download unavailable")
        with self.assertRaisesRegex(EmbeddingError, "download unavailable"):
            SentenceTransformerEncoder(self.settings)
        self.constructor.side_effect = None
        encoder = SentenceTransformerEncoder(self.settings)
        self.model.encode.side_effect = RuntimeError("encoding failed")
        with self.assertRaisesRegex(EmbeddingError, "encoding failed"):
            encoder.encode(["text"], batch_size=1, normalize_embeddings=True)

    def test_import_failure_explains_optional_install(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with self.assertRaisesRegex(EmbeddingError, "pip install"):
                SentenceTransformerEncoder(self.settings)
