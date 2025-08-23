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


from src.utils.mock_data import get_mock_data


import pandas as pd


data = get_mock_data(200, "15m")

df = pd.DataFrame(data)

df.rename(
    columns={0: "time", 1: "open", 2: "high", 3: "low", 4: "close", 5: "volume"},
    inplace=True,
)
