import numpy as np
import numba as nb
from numba.core.types import unicode_type
from numba import types  # 确保导入types

from src.utils.constants import numba_config


# 使用 types 模块定义基本类型（假设 numba_config 已正确设置）
nb_int = numba_config["nb"]["int"]  # e.g., types.int64
nb_float = numba_config["nb"]["float"]  # e.g., types.float64
nb_bool = numba_config["nb"]["bool"]  # e.g., types.boolean

# 定义 numpy 数组类型
tohlcv_np_type = types.DictType(unicode_type, nb_float[:])

# 定义字典类型：使用 types.DictType
param_dict_type = types.DictType(unicode_type, nb_float)

# 定义列表类型：使用 types.ListType
params_list_type = types.ListType(param_dict_type)

# 定义 mapping 字典类型
mapping_dict_type = types.DictType(unicode_type, nb_int[:])

indicators_output_type = types.DictType(unicode_type, nb_float[:])
signal_output_type = types.DictType(unicode_type, nb_bool[:])
backtest_output_type = types.DictType(unicode_type, nb_float[:])
performance_output_type = types.DictType(unicode_type, nb_float)

# 定义返回类型的子项
indicators_list_type = types.ListType(indicators_output_type)
signals_list_type = types.ListType(signal_output_type)
backtest_list_type = types.ListType(backtest_output_type)
performance_list_type = types.ListType(performance_output_type)


# 定义输入签名（参数类型）
input_signature = (
    tohlcv_np_type,
    params_list_type,
    params_list_type,
    tohlcv_np_type,
    params_list_type,
    mapping_dict_type,
    tohlcv_np_type,
    tohlcv_np_type,
    nb_bool,
)

# 定义返回签名（使用 types.Tuple 表示返回的元组类型）
parallel_return_signature = types.Tuple(
    (
        indicators_list_type,
        signals_list_type,
        backtest_list_type,
        performance_list_type,
        indicators_list_type,
    )
)

# 正确构建函数签名
parallel_signature = parallel_return_signature(*input_signature)

indicators_signature = types.void(
    tohlcv_np_type, param_dict_type, indicators_output_type
)

signal_signature = types.void(
    tohlcv_np_type,
    tohlcv_np_type,
    mapping_dict_type,
    indicators_output_type,
    indicators_output_type,
    signal_output_type,
    param_dict_type,
)

backtest_signature = types.void(
    tohlcv_np_type, param_dict_type, signal_output_type, backtest_output_type
)

performance_signature = types.void(
    tohlcv_np_type, param_dict_type, backtest_output_type, performance_output_type
)
