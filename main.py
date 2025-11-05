"""Entry point module for the LISA CLI."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lisa.cli import main


if __name__ == "__main__":
    raise SystemExit(main())





