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
ohlcv_np_type = types.DictType(unicode_type, nb_float[:])
data_mtf_type = types.ListType(ohlcv_np_type)

# 定义字典类型：使用 types.DictType
param_dict_type = types.DictType(unicode_type, nb_float)

# 定义列表类型：使用 types.ListType
params_list_type = types.ListType(param_dict_type)
params_list_mtf_type = types.ListType(types.ListType(param_dict_type))

# 定义 mapping 字典类型
mapping_dict_type = types.DictType(unicode_type, nb_int[:])

indicators_output_type = types.DictType(unicode_type, nb_float[:])
signal_output_type = types.DictType(unicode_type, nb_bool[:])
backtest_output_type = types.DictType(unicode_type, nb_float[:])
performance_output_type = types.DictType(unicode_type, nb_float)

# 定义返回类型的子项
indicators_list_type = types.ListType(indicators_output_type)
indicators_list_mtf_type = types.ListType(types.ListType(indicators_output_type))
signals_list_type = types.ListType(signal_output_type)
backtest_list_type = types.ListType(backtest_output_type)
performance_list_type = types.ListType(performance_output_type)


# 定义输入签名（参数类型）
input_signature = (
    data_mtf_type,  # ohlcv_mtf
    data_mtf_type,  # ohlcv_smoothed_mtf
    mapping_dict_type,  # data_mapping
    params_list_mtf_type,  # indicator_params_mtf
    params_list_type,  # backtest_params
    nb_bool,  # is_only_performance
)

# 定义返回签名（使用 types.Tuple 表示返回的元组类型）
parallel_return_signature = types.Tuple(
    (
        indicators_list_mtf_type,
        signals_list_type,
        backtest_list_type,
        performance_list_type,
    )
)


# 正确构建函数签名
parallel_signature = parallel_return_signature(*input_signature)

indicators_signature = types.void(
    ohlcv_np_type, param_dict_type, indicators_output_type
)

signal_signature = types.void(
    data_mtf_type,  # ohlcv_mtf
    mapping_dict_type,  # data_mapping
    indicators_list_type,  # i_output_mtf
    signal_output_type,  # s_output
    param_dict_type,  # b_params
)

backtest_signature = types.void(
    data_mtf_type,  # ohlcv_mtf
    param_dict_type,  # b_params
    signal_output_type,  # s_output
    backtest_output_type,  # b_output
)

performance_signature = types.void(
    data_mtf_type,  # ohlcv_mtf
    param_dict_type,
    backtest_output_type,
    performance_output_type,
)
