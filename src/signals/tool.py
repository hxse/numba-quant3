import numpy as np
from numba import njit, types
from numba.typed import List, Dict
from numba.core.types import unicode_type


from src.utils.constants import numba_config


enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


# 因为all_indicator_data是元组, 所以用inline避免重复编译
@njit(cache=enable_cache, inline="always")
def populate_indicator_dicts(num, all_indicator_data):
    value_list = List.empty_list(Dict.empty(unicode_type, types.float64))
    optim_list = List.empty_list(Dict.empty(unicode_type, types.float64))

    for i in range(num + 1):
        value_dict = Dict.empty(unicode_type, types.float64)
        optim_dict = Dict.empty(unicode_type, types.float64)
        value_list.append(value_dict)
        optim_list.append(optim_dict)

    for param_tuple in all_indicator_data:
        _n, _n2, _s, _s2, arr = param_tuple

        _value_dict = value_list[_n]
        _optim_dict = optim_list[_n]

        if _s == "" or _s2 == "":
            continue
        elif _s2 == "enable":
            if len(arr) >= 1:
                _value_dict[f"{_s}_{_s2}_{_n2}"] = arr[0]
            if len(arr) >= 2:
                _optim_dict[f"{_s}_{'optim'}_{_n2}"] = arr[1]
        elif len(arr) == 4:
            _value_dict[f"{_s}_{_s2}_{_n2}"] = arr[0]
            _optim_dict[f"{_s}_{_s2}_min_{_n2}"] = arr[1]
            _optim_dict[f"{_s}_{_s2}_max_{_n2}"] = arr[2]
            _optim_dict[f"{_s}_{_s2}_step_{_n2}"] = arr[3]

    return value_list, optim_list
