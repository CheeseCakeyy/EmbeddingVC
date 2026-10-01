"""Command-line entry point."""

import argparse
from pathlib import Path

from . import __version__
from .commands.log import log
from .commands.checkout import checkout
from .commands.diff import diff
from .objects import RepositoryError
from .commands.add import AddError, add
from .commands.embed import EmbedError, embed
from .commands.commit import CommitError, commit
from .commands.status import StatusError, status
from .commands.branch import BranchError, branch
from .commands.config import get as config_get
from .commands.config import set_ as config_set
from .commands.config import show as config_show
from .config import ConfigurationError
from .repository import InitializationError, initialize


def positive_limit(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("limit must be a positive integer")
    return number


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
    add_parser = commands.add_parser("add", help="Track documents and stage deterministic chunks")
    add_parser.add_argument("paths", nargs="+", help="Files or directories inside the repository")
    commands.add_parser("status", help="Show document changes and embedding readiness")
    commands.add_parser("embed", help="Generate embeddings and prepare the candidate snapshot")
    commit_parser = commands.add_parser("commit", help="Save an immutable snapshot and synchronize Chroma")
    commit_parser.add_argument("-m", "--message", required=True, help="Describe this snapshot")
    log_parser = commands.add_parser("log", help="Show committed history, newest first")
    log_parser.add_argument("--limit", type=positive_limit, help="Maximum number of commits")
    diff_parser = commands.add_parser("diff", help="Compare two committed snapshots")
    diff_parser.add_argument("old", help="Old revision")
    diff_parser.add_argument("new", help="New revision")
    checkout_parser = commands.add_parser("checkout", help="Restore a snapshot or repair HEAD synchronization")
    checkout_parser.add_argument("revision", help="Branch, commit revision, or HEAD for sync repair")
    checkout_parser.add_argument("--force", action="store_true", help="Discard staged/configuration changes; preserve source files")
    args = parser.parse_args(argv)
    try:
        if args.command == "checkout":
            print(checkout(args.revision, force=args.force).render())
            return
        if args.command == "log":
            print(log(limit=args.limit))
            return
        if args.command == "diff":
            print(diff(args.old, args.new))
            return
        if args.command == "commit":
            print(commit(message=args.message).render())
            return
        if args.command == "embed":
            print(embed().render())
            return
        if args.command == "status":
            print(status())
            return
        if args.command == "add":
            print(add(args.paths).render())
            return
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
    except (RepositoryError, InitializationError, BranchError, ConfigurationError, AddError, StatusError, EmbedError, CommitError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"EmbeddingVC repository initialized at {root}.\n")
    print("Next steps:\n1. Add documents to data/\n2. Review embeddingvc.yaml")
    print("3. Run embeddingvc add ./data, then embeddingvc status")
