"""Checkout committed configuration, index and stored vectors."""
from dataclasses import dataclass
from pathlib import Path

from ..objects import RepositoryError, find_repository
from ..repository import repository_lock
from ..restore_engine import restore


class CheckoutError(RepositoryError):
    """A checkout could not safely complete."""


@dataclass(frozen=True)
class Result:
    commit: str
    branch: str | None
    records: int
    repaired: bool

    def render(self):
        if self.repaired:
            return (f"Synchronized HEAD {self.commit[:12]}: {self.records} records\n"
                    "Working configuration, index and source files were preserved.")
        label = f"branch {self.branch}" if self.branch else "detached HEAD"
        text = (f"Checked out {self.commit[:12]} ({label})\n"
                f"Restored configuration and index; Chroma synchronized: {self.records} records\n"
                "Source files were preserved. Run embeddingvc status.")
        if not self.branch:
            text += "\nCreate and check out a branch before committing."
        return text


def checkout(revision: str, directory: Path | str = ".", *, force=False, adapter_factory=None):
    try:
        root = find_repository(Path(directory))
        with repository_lock(root):
            return Result(*restore(root, revision, force=force, adapter_factory=adapter_factory))
    except Exception as exc:
        raise CheckoutError(str(exc)) from exc
