import numpy as np
import numba as nb
from numba import njit, prange
from numba.core import types
from numba.typed import Dict


numba_config = {}


def get_numba_dtypes(enable64: bool):
    np_int_type = np.int64 if enable64 else np.int32
    np_float_type = np.float64 if enable64 else np.float32
    np_bool_type = np.bool_
    nb_int_type = nb.int64 if enable64 else nb.int32
    nb_float_type = nb.float64 if enable64 else nb.float32
    nb_bool_type = nb.boolean
    dtype_dict = {
        "np": {"int": np_int_type, "float": np_float_type, "bool": np_bool_type},
        "nb": {"int": nb_int_type, "float": nb_float_type, "bool": nb_bool_type},
    }
    return dtype_dict


def set_numba_dtypes(cache: bool, enable64: bool):
    global numba_config, np_int, np_float, np_bool, nb_int, nb_float, nb_bool

    numba_config["cache"] = cache
    numba_config["enable64"] = enable64

    dtype_dict = get_numba_dtypes(enable64)
    for k, v in dtype_dict.items():
        numba_config[k] = v
