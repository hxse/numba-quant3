import numpy as np
import numba as nb
from numba.core.types import unicode_type
from numba import types  # 确保导入types

from src.utils.constants import numba_config

cache = numba_config["cache"]

# 使用 types 模块定义基本类型（假设 numba_config 已正确设置）
nb_int = numba_config["nb"]["int"]  # e.g., types.int64
nb_float = numba_config["nb"]["float"]  # e.g., types.float64
nb_bool = numba_config["nb"]["bool"]  # e.g., types.boolean

# 定义 numpy 数组类型
tohlcv_np_type = nb_float[:, :]

# 定义字典类型：使用 types.DictType
param_dict_type = types.DictType(unicode_type, nb_float)

# 定义列表类型：使用 types.ListType
params_list_type = types.ListType(param_dict_type)

# 定义 mapping 字典类型
mapping_dict_type = types.DictType(unicode_type, nb_int[:])

# 定义返回类型的子项
indicators_list_type = types.ListType(types.DictType(unicode_type, nb_float[:]))
signals_list_type = types.ListType(types.DictType(unicode_type, nb_bool[:]))
backtest_list_type = types.ListType(types.DictType(unicode_type, nb_float[:]))
performance_list_type = types.ListType(types.DictType(unicode_type, nb_float))

# 定义输入签名（参数类型）
input_signature = (
    tohlcv_np_type,
    params_list_type,
    params_list_type,
    types.Optional(tohlcv_np_type),
    types.Optional(params_list_type),
    types.Optional(mapping_dict_type),
)

# 定义返回签名（使用 types.Tuple 表示返回的元组类型）
return_signature = types.Tuple(
    (
        indicators_list_type,
        signals_list_type,
        backtest_list_type,
        performance_list_type,
        indicators_list_type,
    )
)

# 正确构建函数签名
signature = return_signature(*input_signature)
