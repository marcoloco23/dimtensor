"""Entry point for python -m dimtensor."""

from __future__ import annotations

import sys


def main() -> int:
    """Main entry point for dimtensor CLI."""
    if len(sys.argv) < 2:
        print("Usage: python -m dimtensor <command> [options]")
        print()
        print("Commands:")
        print("  lint    Check files for dimensional issues")
        print("  info    Show information about dimtensor")
        print()
        print("Run 'python -m dimtensor <command> --help' for more information.")
        return 0

    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # Remove 'dimtensor' from argv

    if command == "lint":
        from dimtensor.cli.lint import main as lint_main

        return lint_main()

    elif command == "info":
        from dimtensor import __version__
        from dimtensor._rust import HAS_RUST_BACKEND

        print(f"dimtensor {__version__}")
        print(f"Rust backend: {'available' if HAS_RUST_BACKEND else 'not available'}")
        return 0

    elif command in ("--help", "-h"):
        return main.__wrapped__() if hasattr(main, "__wrapped__") else 0

    else:
        print(f"Unknown command: {command}")
        print("Run 'python -m dimtensor --help' for available commands.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
