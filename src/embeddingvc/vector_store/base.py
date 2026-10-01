"""Minimal replacement contract shared with checkout."""

from typing import Protocol


class VectorStore(Protocol):
    def replace(self, commit: str, records: list[dict], dimension: int) -> str:
        """Build and verify a new collection; return its physical name.

        Never alter a previously published collection. The caller atomically
        publishes the returned name in sync.json only after success.
        """
        ...
