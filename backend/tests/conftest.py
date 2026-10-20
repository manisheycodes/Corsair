"""Keep API tests from adding projects and artifacts to a developer's local data."""

import os
from pathlib import Path


os.environ["AUTOBUILD_DATA_DIR"] = str(
    Path(__file__).resolve().parents[1] / "data" / "test-suite"
)
