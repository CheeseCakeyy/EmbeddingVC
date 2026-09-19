"""Command-line entry point."""

import argparse
from pathlib import Path

from . import __version__
from .commands.add import AddError, add
from .repository import InitializationError, initialize


def app(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="embeddingvc")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Initialize an embedding repository")
    init.add_argument("directory", nargs="?", type=Path, default=Path("."))
    init.add_argument("--force", action="store_true", help="Replace existing README and config; never reset history")
    stage = commands.add_parser("add", help="Track documents and stage their chunks")
    stage.add_argument("paths", nargs="+", help="Files or directories inside the repository")
    args = parser.parse_args(argv)
    if args.command == "add":
        _add(parser, args)
    else:
        _init(parser, args)


def _init(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        root = initialize(args.directory, force=args.force)
    except (InitializationError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"EmbeddingVC repository initialized at {root}.\n")
    print("Next steps:\n1. Add documents to data/\n2. Review embeddingvc.yaml")
    print("3. Next planned command: embeddingvc status (not implemented yet)")


def _add(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        result = add(args.paths)
    except (AddError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(result.render())
