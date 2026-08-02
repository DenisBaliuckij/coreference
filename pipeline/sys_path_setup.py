from __future__ import annotations

import os
import sys
from pathlib import Path


def add_text_corpuses_processing_to_path() -> Path:
    """Add the sibling text-corpuses-processing repo's dags/ dir to sys.path.

    No existence check: if the directory is missing, later imports of the
    reused modules fail with the standard ModuleNotFoundError, which is more
    informative than a raise here and keeps this function safe to call
    unconditionally in tests that mock the reused modules via sys.modules.
    """
    override = os.environ.get("TEXT_CORPUSES_PROCESSING_DAGS")
    if override:
        dags_dir = Path(override)
    else:
        dags_dir = Path(__file__).resolve().parents[1] / "text-corpuses-processing" / "dags"

    dags_str = str(dags_dir)
    if dags_str not in sys.path:
        sys.path.insert(0, dags_str)
    return dags_dir
