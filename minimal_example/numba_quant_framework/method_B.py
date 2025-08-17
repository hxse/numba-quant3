import numpy as np
from numba import njit, prange
from numba.core import types
from numba.typed import Dict

type_dict = {
    # NumPy 的数据类型
    "np": {
        "64": {
            "float": np.float64,
            "int": np.int64,
        },
        "32": {
            "float": np.float32,
            "int": np.int32,
        },
    },
    # Numba 的数据类型
    "nb": {
        "64": {
            "float": types.float64,
            "int": types.int64,
        },
        "32": {
            "float": types.float32,
            "int": types.int32,
        },
    },
}

np_float = type_dict["np"]["64"]["float"]
nb_float = type_dict["nb"]["64"]["float"]

np.random.seed(42)

data_count = 3
tohlcv = Dict.empty(
    key_type=types.unicode_type,
    value_type=nb_float[:],
)
tohlcv["close"] = np.random.rand(data_count).astype(np_float)

# 方案二的参数结构
params_dict = Dict.empty(
    key_type=types.unicode_type,
    value_type=nb_float[:],
)
params_dict["sma_period"] = np.asarray([14, 100]).astype(np_float)
params_dict["bbands_period"] = np.asarray([20, 200]).astype(np_float)


@njit(parallel=True)
def vectorized_move(tohlcv, params_dict):
    # 预先分配结果数组，行数=参数组合数，列数=数据点数
    num_params = len(params_dict["sma_period"])
    num_data = len(tohlcv["close"])

    # 返回一个字典，键为指标名，值为二维结果数组
    res_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float[:, :],
    )

    # 预分配两个二维数组来存储结果
    res_sma = np.empty((num_params, num_data), dtype=np_float)
    res_bbands = np.empty((num_params, num_data), dtype=np_float)

    close = tohlcv["close"]
    sma_periods = params_dict["sma_period"]
    bbands_periods = params_dict["bbands_period"]

    for i in prange(num_params):
        # 对每个参数组合进行计算，并将结果存入对应的二维数组行中
        res_sma[i] = close + sma_periods[i]
        res_bbands[i] = close + bbands_periods[i]

    res_dict["sma"] = res_sma
    res_dict["bbands"] = res_bbands

    return res_dict


print("tohlcv", tohlcv)
print("params_dict", params_dict)

# 运行并打印结果
res = vectorized_move(tohlcv, params_dict)

print(res)
print(res["sma"])
print(res["bbands"])
