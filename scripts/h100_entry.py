"""Portable entry point for the measured eight-GPU recipe."""
import os
from pathlib import Path
import sys

# A baseline checkout can use this same adapter without copying it into that tree.
source = Path(os.environ.get("R3_SOURCE_TREE", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(source))
import train.online_rl as entry
from h100_adapt import apply

if __name__ == "__main__":
    if "--log_dir" not in sys.argv:
        raise SystemExit("--log_dir is required")
    log_dir = sys.argv[sys.argv.index("--log_dir") + 1]
    apply(entry, log_dir)
    entry.main()
