"""Command-line entry point."""

import argparse
from pathlib import Path

from . import __version__
from .repository import InitializationError, initialize


def app(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="embeddingvc")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Initialize an embedding repository")
    init.add_argument("directory", nargs="?", type=Path, default=Path("."))
    init.add_argument("--force", action="store_true", help="Replace existing README and config; never reset history")
    args = parser.parse_args(argv)
    try:
        root = initialize(args.directory, force=args.force)
    except (InitializationError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"EmbeddingVC repository initialized at {root}.\n")
    print("Next steps:\n1. Add documents to data/\n2. Review embeddingvc.yaml")
    print("3. Next planned command: embeddingvc status (not implemented yet)")
