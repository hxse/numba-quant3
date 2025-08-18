import sys
from pathlib import Path

root_path = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()),
    None,
)
if root_path:
    sys.path.insert(0, str(root_path))

from src.utils.constants import numba_config, set_numba_dtypes


set_numba_dtypes(numba_config, cache=True, enable64=True)
