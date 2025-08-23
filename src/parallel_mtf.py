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


from parallel_mtf_signature import signature

cache = numba_config["cache"]

nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]

print("parallel_entry cache", cache)


@njit(cache=cache)
def init_output_all(params_count):
    # 使用显式类型创建 Typed List
    indicators_output_list = List.empty_list(
        Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
    )
    signals_output_list = List.empty_list(
        Dict.empty(key_type=types.unicode_type, value_type=nb_bool[:])
    )
    backtest_output_list = List.empty_list(
        Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
    )
    performance_output_list = List.empty_list(
        Dict.empty(key_type=types.unicode_type, value_type=nb_float)
    )
    indicators_output_list_mtf = List.empty_list(
        Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
    )

    # 预填充列表
    for _ in range(params_count):
        # 填充一个空的字典
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


@njit(signature, parallel=True, cache=cache)
def run_parallel_mtf(
    tohlcv,
    indicator_params_list,
    backtest_params_list,
    tohlcv_mtf,
    indicator_params_list_mtf,
    mapping_mtf,
    tohlcv_smoothed,
    tohlcv_mtf_smoothed,
):
    """
    并发200配置和4万数据,如果加上njit,缓存,parallel,这个是0.1391 秒,0.1346 秒,0.1246 秒
    并发1  配置和4万数据,如果加上njit,缓存,parallel,这个是0.0499 秒,0.0488 秒,0.0462 秒
    """
    assert len(indicator_params_list) == len(backtest_params_list), (
        "参数组合数量需要相等"
    )

    _tohlcv = tohlcv if len(tohlcv_smoothed) == 0 else tohlcv_smoothed
    _tohlcv_mtf = tohlcv_mtf if len(tohlcv_mtf_smoothed) == 0 else tohlcv_mtf_smoothed

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
        indicator_params_mtf = indicator_params_list_mtf[_i]

        indicator_output = indicators_output_list[_i]
        signal_output = signals_output_list[_i]
        backtest_output = backtest_output_list[_i]
        performance_output = performance_output_list[_i]
        indicators_output_mtf = indicators_output_list_mtf[_i]

        calc_indicators(_tohlcv, indicator_params, indicator_output)

        if len(tohlcv_mtf) > 0:
            calc_indicators(_tohlcv_mtf, indicator_params_mtf, indicators_output_mtf)

        calc_signal(
            _tohlcv,
            _tohlcv_mtf,
            mapping_mtf,
            indicator_output,
            indicators_output_mtf,
            signal_output,
            backtest_params,
        )

        calc_backtest(tohlcv, backtest_params, signal_output, backtest_output)

        calc_performance(tohlcv, backtest_params, backtest_output, performance_output)

    return (
        indicators_output_list,
        signals_output_list,
        backtest_output_list,
        performance_output_list,
        indicators_output_list_mtf,
    )
