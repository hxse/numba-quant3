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

print("parallel_entry cache", cache)


@njit(cache=cache)
def init_tohlcv(np_data):
    tohlcv = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float[:],
    )
    if np_data is None:
        return tohlcv
    assert np_data.shape[1] >= 6, "tohlcv数据列数不足"
    tohlcv["time"] = np_data[:, 0]
    tohlcv["open"] = np_data[:, 1]
    tohlcv["high"] = np_data[:, 2]
    tohlcv["low"] = np_data[:, 3]
    tohlcv["close"] = np_data[:, 4]
    tohlcv["volume"] = np_data[:, 5]
    return tohlcv


@njit(cache=cache)
def init_output_all(params_count):
    # 在 @njit 之外进行预分配
    # 1. 创建一个 Typed List，元素类型是 Dict
    indicators_output_list = List()
    signals_output_list = List()
    backtest_output_list = List()
    performance_output_list = List()
    indicators_output_list_mtf = List()
    # 2. 预填充 List，使其具有正确的长度
    for _ in range(params_count):
        # 填充一个空的字典，这样 List 就有了确定的元素类型和长度
        indicators_output_list.append(
            Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
        )
        signals_output_list.append(
            Dict.empty(key_type=types.unicode_type, value_type=nb_bool[:])
        )
        backtest_output_list.append(
            Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
        )
        performance_output_list.append(
            Dict.empty(key_type=types.unicode_type, value_type=nb_float)
        )
        indicators_output_list_mtf.append(
            Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
        )
    return (
        indicators_output_list,
        signals_output_list,
        backtest_output_list,
        performance_output_list,
        indicators_output_list_mtf,
    )


@njit(parallel=True, cache=cache)
def run_parallel_mtf(
    tohlcv_np,
    indicator_params_list,
    backtest_params_list,
    tohlcv_np_mtf=None,
    indicator_params_list_mtf=None,
    mapping_mtf=None,
):
    """
    并发200配置和4万数据,如果加上njit,缓存,parallel,这个是0.1391 秒,0.1346 秒,0.1246 秒
    并发1  配置和4万数据,如果加上njit,缓存,parallel,这个是0.0499 秒,0.0488 秒,0.0462 秒
    """
    assert len(indicator_params_list) == len(backtest_params_list), (
        "参数组合数量需要相等"
    )

    tohlcv = init_tohlcv(tohlcv_np)
    tohlcv_mtf = init_tohlcv(tohlcv_np_mtf)

    close = tohlcv["close"]

    params_count = len(indicator_params_list)

    (
        indicators_output_list,
        signals_output_list,
        backtest_output_list,
        performance_output_list,
        indicators_output_list_mtf,
    ) = init_output_all(params_count)

    for i in prange(params_count):
        _i = nb_int(i)
        indicator_params = indicator_params_list[_i]
        backtest_params = backtest_params_list[_i]

        indicator_output = indicators_output_list[_i]

        signal_output = signals_output_list[_i]
        backtest_output = backtest_output_list[_i]
        performance_output = performance_output_list[_i]

        calc_indicators(indicator_output, close, indicator_params)

        indicators_output_mtf = indicators_output_list_mtf[_i]

        if tohlcv_np_mtf is not None and indicator_params_list_mtf is not None:
            indicator_params_mtf = indicator_params_list_mtf[_i]
            close_mtf = tohlcv_mtf["close"]
            calc_indicators(indicators_output_mtf, close_mtf, indicator_params_mtf)

        calc_signal(
            signal_output,
            indicator_output,
            indicators_output_mtf,
            mapping_mtf,
            close,
            backtest_params,
        )

        calc_backtest(backtest_output, signal_output, close, backtest_params)

        calc_performance(performance_output, backtest_output, close, backtest_params)

    return (
        indicators_output_list,
        signals_output_list,
        backtest_output_list,
        performance_output_list,
        indicators_output_list_mtf,
    )
