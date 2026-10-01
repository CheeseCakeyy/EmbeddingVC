"""Lazy Sentence Transformers adapter with pinned, reproducible provenance."""

import re
from importlib.metadata import version

from .hashing import hash_payload, hash_text
from .object_store import validate_vector


class EmbeddingError(Exception):
    """The configured model cannot produce valid vectors."""


def validate_revision(embedding: dict) -> None:
    revision = embedding.get("revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise EmbeddingError(
            "Pin embedding.revision to a full 40-character model commit SHA. "
            "Run embeddingvc config set model_revision <exact-model-commit>."
        )


def effective_configuration(settings, staging: dict, dimension: int, provenance: dict) -> dict:
    return {
        "schema": 1,
        "pipeline": staging,
        "model": settings.embedding["model"],
        "revision": settings.embedding["revision"],
        "pooling": settings.embedding.get("pooling", "model-default"),
        "dimension": dimension,
        "normalize_embeddings": settings.embedding["normalize_embeddings"],
        "provenance": provenance,
    }


class SentenceTransformerEncoder:
    """Load only on embed; model weights stay in the library's normal cache."""

    def __init__(self, embedding: dict):
        validate_revision(embedding)
        try:
            from sentence_transformers import SentenceTransformer, models
        except Exception as exc:
            raise EmbeddingError(
                'Cannot import Sentence Transformers. Install the embedding dependencies with: '
                'pip install -e ".[embeddings]"'
            ) from exc
        try:
            self.model = SentenceTransformer(embedding["model"], revision=embedding["revision"],
                                             trust_remote_code=False, device="cpu")
            # encode(precision="float32") controls output quantization; it does
            # not guarantee the model weights use the recorded compute dtype.
            self.model.float()
            mode = embedding.get("pooling", "model-default")
            poolings = [(name, module) for name, module in self.model._modules.items()
                        if isinstance(module, models.Pooling)]
            if mode != "model-default":
                if len(poolings) != 1:
                    raise EmbeddingError("Custom pooling requires a model with exactly one Pooling module")
                name, previous = poolings[0]
                width = previous.get_config_dict()["word_embedding_dimension"]
                self.model._modules[name] = models.Pooling(
                    width, pooling_mode=mode, include_prompt=previous.include_prompt)
            dimension = self.model.get_sentence_embedding_dimension()
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
                raise EmbeddingError("Model did not report a positive embedding dimension")
            expected = embedding.get("dimension")
            if expected is not None and dimension != expected:
                raise EmbeddingError(f"Model dimension mismatch: configured {expected}, model reports {dimension}")
            self.dimension = dimension
            first = self.model._first_module()
            model_config = first.auto_model.config
            resolved = getattr(model_config, "_commit_hash", None)
            if resolved != embedding["revision"]:
                raise EmbeddingError("Loaded model revision does not match the pinned commit")
            tokenizer = self.model.tokenizer
            # Serialize the tokenizer itself when possible: vocab alone omits
            # normalizer, pre-tokenizer, special-token and truncation behavior.
            backend = getattr(tokenizer, "backend_tokenizer", None)
            tokenizer_hash = (hash_text(backend.to_str()) if backend is not None
                              else hash_payload(tokenizer.get_vocab()))
            modules = []
            for module in self.model._modules.values():
                get_config = getattr(module, "get_config_dict", None)
                modules.append({"class": f"{type(module).__module__}.{type(module).__name__}",
                                "config": get_config() if callable(get_config) else {}})
            self.provenance = {
                "provider": "sentence-transformers",
                "model": embedding["model"],
                "model_revision": resolved,
                "model_config_hash": hash_payload(model_config.to_dict()),
                "tokenizer": {"class": type(tokenizer).__name__, "revision": resolved,
                              "hash": tokenizer_hash, "max_length": tokenizer.model_max_length,
                              "padding_side": tokenizer.padding_side,
                              "truncation_side": tokenizer.truncation_side},
                "modules": modules,
                "max_seq_length": self.model.max_seq_length,
                "prompts": self.model.prompts,
                "default_prompt_name": self.model.default_prompt_name,
                "libraries": {name: version(name) for name in
                              ("sentence-transformers", "transformers", "torch", "tokenizers")},
                "device": "cpu",
                "precision": "float32",
            }
            # Fail before encoding if provenance cannot be stored canonically.
            hash_payload(self.provenance)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Cannot load pinned model or tokenizer: {exc}") from exc

    def encode(self, texts: list[str], *, batch_size: int, normalize_embeddings: bool):
        try:
            return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                                     convert_to_numpy=True, normalize_embeddings=normalize_embeddings,
                                     precision="float32").tolist()
        except Exception as exc:
            raise EmbeddingError(f"Model encoding failed: {exc}") from exc


def encode_batch(encoder, texts: list[str], *, batch_size: int, normalize_embeddings: bool) -> list[list[float]]:
    try:
        vectors = encoder.encode(texts, batch_size=batch_size, normalize_embeddings=normalize_embeddings)
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingError("Model returned the wrong number of vectors")
        return [validate_vector(vector, encoder.dimension) for vector in vectors]
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Invalid model output: {exc}") from exc
