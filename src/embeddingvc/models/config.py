"""Small schema model used by the configuration command."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Configuration:
    """A validated repository configuration represented as nested data."""

    data: dict
    effective_hash: str
