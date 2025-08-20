import numpy as np
import numba as nb
from numba import prange

cache = True

print("numba cache:", cache)


@nb.njit(cache=cache, parallel=True)
def sub_func(data, flag):
    # Numba will generate different specializations for (int64[:], int) and (int64[:], float)
    if flag > 0:
        print("sub_func: Positive value passed, performing array operation.")
        for i in prange(len(data)):
            data[i] = data[i] * flag
    else:
        print("sub_func: Non-positive value passed, performing other operation.")
        # Ensure the operation is type-compatible here
        data[0] = 999
    return data


@nb.njit(cache=cache, parallel=True)
def shell_func(data, flag):
    return sub_func(data, flag)


if __name__ == "__main__":
    """
    https://github.com/numba/numba/issues/10184
    """
    arr = np.ones(5, dtype=np.int64)

    # --- Nested Call Test ---
    # Calls the parallel sub_func from a non-parallel njit function shell_func
    # The first run will compile and cache, but subsequent runs may behave abnormally due to cache conflicts
    print("--- Nested Call (shell_func calling sub_func) ---")
    try:
        # Fails due to a Numba cache/parallel conflict.
        result_nested = shell_func(arr, 2)
        # result_nested = sub_func(arr, 2)  # successful
        print("Result:", result_nested)
    except Exception as e:
        print("Error:", e)

    print("\n" + "=" * 40 + "\n")
