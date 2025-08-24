from src.utils.constants import numba_config, set_numba_dtypes


set_numba_dtypes(numba_config, cache=True, enable64=True)
