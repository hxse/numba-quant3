import numpy as np
import numba as nb
from numba import njit
from numba.core import types
from numba.typed import Dict, List

# 从全局配置中获取 Numba 类型
from src.utils.constants import numba_config


cache = numba_config["cache"]

nb_float = numba_config["nb"]["float"]
np_float = numba_config["np"]["float"]


# 将此函数用 @njit 装饰，使其能被 Numba 编译
@njit(cache=cache)
def get_indicator_params():
    params = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float,
    )
    params["sma_period"] = float(14)
    params["sma2_period"] = float(14)
    params["bbands_period"] = float(14)
    params["bbands_std_mult"] = float(2.0)
    return params


# 将此函数用 @njit 装饰，使其能被 Numba 编译
@njit(cache=cache)
def get_backtest_params():
    params = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float,
    )
    params["signal_select"] = float(0)
    params["atr_sl_mult"] = float(2.0)
    return params


# 将主函数用 @njit 装饰
@njit(cache=cache)
def get_params_template(params_count):
    """
    生成回测所需的数据和参数。
    整个函数都将在 Numba 中执行，从而实现高性能。
    """
    assert params_count > 0, "参数组合数量必须大于0"
    # 在 JIT 编译的循环中调用 JIT 编译的函数
    indicator_params_list = List.empty_list(
        Dict.empty(
            key_type=types.unicode_type,
            value_type=nb_float,
        )
    )
    backtest_params_list = List.empty_list(
        Dict.empty(
            key_type=types.unicode_type,
            value_type=nb_float,
        )
    )

    for n in range(params_count):
        indicator_params_list.append(get_indicator_params())
        backtest_params_list.append(get_backtest_params())

    return (indicator_params_list, backtest_params_list)
