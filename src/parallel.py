import numpy as np
import numba as nb
from numba import njit, prange
from numba.core import types
from numba.typed import Dict, List

from src.utils.constants import numba_config


from src.indicators.calculate_indicators import calc_indicators
from src.signals.calculate_signal import calc_signal
from src.backtest.calculate_backtest import calc_backtest
from src.backtest.calculate_performance import calc_performance


cache = numba_config["cache"]

nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]

print("parallel_entry", cache)


@njit(parallel=True, cache=cache)
def parallel_entry(tohlcv_np, indicator_params_list, backtest_params_list):
    assert len(indicator_params_list) == len(backtest_params_list), (
        "参数组合数量需要相等"
    )

    tohlcv = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float[:],
    )
    tohlcv["close"] = tohlcv_np[:, 4]

    # 在 @njit 之外进行预分配
    # 1. 创建一个 Typed List，元素类型是 Dict
    indicators_list = List()
    signals_list = List()
    backtest_list = List()
    performance_list = List()
    # 2. 预填充 List，使其具有正确的长度
    for _ in range(len(indicator_params_list)):
        # 填充一个空的字典，这样 List 就有了确定的元素类型和长度
        indicators_list.append(
            Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
        )
        signals_list.append(
            Dict.empty(key_type=types.unicode_type, value_type=nb_bool[:])
        )
        backtest_list.append(
            Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
        )
        performance_list.append(
            Dict.empty(key_type=types.unicode_type, value_type=nb_float)
        )

    for i in prange(len(indicator_params_list)):
        _i = nb_int(i)
        indicator_params = indicator_params_list[_i]
        backtest_params = backtest_params_list[_i]

        close = tohlcv["close"]
        indicator_item = indicators_list[_i]
        signal_item = signals_list[_i]
        backtest_item = backtest_list[_i]
        performance_item = performance_list[_i]

        calc_indicators(indicator_item, close, indicator_params)
        calc_signal(signal_item, indicator_item, close, backtest_params)
        calc_backtest(backtest_item, signal_item, close, backtest_params)
        calc_performance(performance_item, backtest_item, close, backtest_params)

    return indicators_list, signals_list, backtest_list, performance_list
