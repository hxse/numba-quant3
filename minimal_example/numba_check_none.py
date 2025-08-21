import numpy as np
import numba as nb
from numba import njit, prange, types
from numba.typed import Dict, List
from numba.core.types import Optional, unicode_type

numba_config = {
    "cache": False,
    "nb": {"int": types.int64, "float": types.float64, "bool": types.boolean},
}
cache = numba_config["cache"]
nb_float = numba_config["nb"]["float"]
nb_float_array = nb_float[:]

item_dict_type = types.DictType(unicode_type, nb_float)
indicators_list_type = types.ListType(types.DictType(unicode_type, nb_float_array))
params_list_type = types.ListType(item_dict_type)

input_signature = (
    types.int64,  # params_count
    types.Optional(params_list_type),  # optional_params
)

return_signature = nb_float[:]

signature = return_signature(*input_signature)


@njit(cache=cache)
def process_data(data, close):
    if data is None:
        print("Data is None, doing nothing.")
        return

    for item in data:
        if "value" in item:
            print(f"Processing value: {item['value'] * close[0]}")


@njit(cache=cache)
def check_none(optional_params):
    if optional_params is None:
        optional_params = List.empty_list(item_dict_type)
    return optional_params


def run_parallel_example(params_count, optional_params=None):
    # optional_params = check_none(optional_params)

    close = np.array([10.0, 20.0, 30.0], dtype=nb_float)

    for i in prange(params_count):
        process_data(optional_params, close)

    return close


def run_parallel_example2(params_count, optional_params=None):
    # This one line is the only difference with run_parallel_example
    optional_params = check_none(optional_params)

    close = np.array([10.0, 20.0, 30.0], dtype=nb_float)

    for i in prange(params_count):
        process_data(optional_params, close)

    return close


if __name__ == "__main__":
    for _f in [run_parallel_example, run_parallel_example2]:
        if _f == run_parallel_example:
            print("\n ###### run_parallel_example ######")
        else:
            print("\n ###### run_parallel_example2 ######")

        print("\n--- mode 1 add signature, pass None ---")
        try:
            func = njit(signature, parallel=True, cache=True)(_f)
            func(2, optional_params=None)
            print("mode 1 successful")
        except Exception as e:
            print("mode 1 error")
            print(e)

        print("\n--- 2 no signature, pass None ---")

        try:
            func = njit(parallel=True, cache=True)(_f)
            func(2, optional_params=None)
            print("mode 2 successful")
        except Exception as e:
            print("mode 2 error")
            print(e)

        print("\n" + "-" * 30 + "\n")

        print("\n--- 3 add signature, pass test_params ---")

        try:
            test_params = List.empty_list(item_dict_type)
            d = Dict.empty(key_type=unicode_type, value_type=nb_float)
            d["value"] = 5.0
            test_params.append(d)

            func = njit(signature, parallel=True, cache=True)(_f)
            func(2, optional_params=test_params)
            print("mode 3 successful")
        except Exception as e:
            print("mode 3 error")
            print(e)

        print("\n--- 4 no signature, pass test_params ---")

        try:
            test_params = List.empty_list(item_dict_type)
            d = Dict.empty(key_type=unicode_type, value_type=nb_float)
            d["value"] = 5.0
            test_params.append(d)

            func = njit(parallel=True, cache=True)(_f)
            func(2, optional_params=test_params)
            print("mode 4 successful")
        except Exception as e:
            print("mode 4 error")
            print(e)
