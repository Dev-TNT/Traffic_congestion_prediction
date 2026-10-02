"""Compatibility entry point; editable training code is training_model.Py."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "training_model.Py"), run_name="__main__")
