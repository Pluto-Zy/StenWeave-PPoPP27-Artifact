"""Command-line entry point for the Box2D1R quick validation."""

import subprocess
import sys

from quickcheck.cli import main

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Quickcheck failed: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Quickcheck interrupted", file=sys.stderr)
        sys.exit(130)
