"""Command-line entry point."""

import argparse
from pathlib import Path

from . import __version__
from .commands.branch import BranchError, branch
from .commands.config import get as config_get
from .commands.config import set_ as config_set
from .commands.config import show as config_show
from .config import ConfigurationError
from .repository import InitializationError, initialize


def app(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="embeddingvc")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Initialize an embedding repository")
    init.add_argument("directory", nargs="?", type=Path, default=Path("."))
    init.add_argument("--force", action="store_true", help="Replace existing README and config; never reset history")
    branch_parser = commands.add_parser("branch", help="List or create local branches")
    branch_parser.add_argument("name", nargs="?", help="New branch name")
    branch_parser.add_argument("start", nargs="?", help="Existing commit or unique commit prefix")
    config_parser = commands.add_parser("config", help="Show or update repository configuration")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    config_commands.add_parser("show", help="Print validated configuration")
    config_get_parser = config_commands.add_parser("get", help="Read a configuration value")
    config_get_parser.add_argument("key")
    config_set_parser = config_commands.add_parser("set", help="Validate and update a configuration value")
    config_set_parser.add_argument("key")
    config_set_parser.add_argument("value")
    args = parser.parse_args(argv)
    try:
        if args.command == "branch":
            print(branch(name=args.name, start=args.start))
            return
        if args.command == "config":
            if args.config_command == "show":
                print(config_show())
            elif args.config_command == "get":
                print(config_get(Path("."), args.key))
            else:
                print(config_set(Path("."), args.key, args.value))
            return
        root = initialize(args.directory, force=args.force)
    except (InitializationError, BranchError, ConfigurationError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"EmbeddingVC repository initialized at {root}.\n")
    print("Next steps:\n1. Add documents to data/\n2. Review embeddingvc.yaml")
    print("3. Next planned command: embeddingvc status (not implemented yet)")
