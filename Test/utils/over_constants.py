from src.utils.constants import numba_config, set_numba_dtypes


set_numba_dtypes(numba_config, enable_cache=True, enable64=True)
