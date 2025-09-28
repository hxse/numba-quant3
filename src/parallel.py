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


from src.parallel_signature import parallel_signature


enable_cache = numba_config["enable_cache"]


nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


# print("parallel_entry cache", enable_cache)


@njit(cache=enable_cache)
def init_output_all(params_count, mtf_count, enable_fill):
    # 使用显式类型创建 Typed List

    indicators_inner = List.empty_list(
        Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
    )
    indicators_output_mtf = List.empty_list(indicators_inner)
    signals_output = List.empty_list(
        Dict.empty(key_type=types.unicode_type, value_type=nb_bool[:])
    )
    backtest_output = List.empty_list(
        Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
    )
    performance_output = List.empty_list(
        Dict.empty(key_type=types.unicode_type, value_type=nb_float)
    )

    if enable_fill:
        for _ in range(params_count):
            _indicators_inner = List.empty_list(
                Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
            )
            for _ in range(mtf_count):
                _indicators_inner.append(
                    Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
                )

            indicators_output_mtf.append(_indicators_inner)
            signals_output.append(
                Dict.empty(key_type=types.unicode_type, value_type=nb_bool[:])
            )
            backtest_output.append(
                Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
            )
            performance_output.append(
                Dict.empty(key_type=types.unicode_type, value_type=nb_float)
            )
    return (
        indicators_output_mtf,
        signals_output,
        backtest_output,
        performance_output,
    )


# 这是一个用于清空单个回测结果字典的工具函数。
@njit(cache=enable_cache)
def clear_list_element_at_index(
    i,
    indicators_output_mtf,
    signals_output,
    backtest_output,
):
    """
    根据索引i，清空指定列表中对应位置的字典，
    用一个新的空字典替换。
    """
    i_output_mtf = indicators_output_mtf[i]
    for m in range(len(i_output_mtf)):
        i_output_mtf[m] = Dict.empty(
            key_type=types.unicode_type, value_type=types.float64[:]
        )
    signals_output[i] = Dict.empty(
        key_type=types.unicode_type, value_type=types.boolean[:]
    )
    backtest_output[i] = Dict.empty(
        key_type=types.unicode_type, value_type=types.float64[:]
    )


@njit(parallel_signature, parallel=True, cache=enable_cache)
def run_parallel(
    ohlcv_mtf,
    ohlcv_smoothed_mtf,
    data_mapping,
    indicator_params_mtf,
    backtest_params,
    is_only_performance,
):
    """
    并发200配置和4万数据,如果加上njit,缓存,parallel,这个是0.1391 秒,0.1346 秒,0.1246 秒
    并发1  配置和4万数据,如果加上njit,缓存,parallel,这个是0.0499 秒,0.0488 秒,0.0462 秒
    """
    assert len(indicator_params_mtf) == len(backtest_params), "参数组合数量需要相等"

    _ohlcv_mtf = ohlcv_mtf if len(ohlcv_smoothed_mtf) == 0 else ohlcv_smoothed_mtf

    mtf_count = len(ohlcv_mtf)
    params_count = len(indicator_params_mtf)

    (
        indicators_output_mtf,
        signals_output,
        backtest_output,
        performance_output,
    ) = init_output_all(params_count, mtf_count, True)

    for i in prange(params_count):
        _i = nb_int(i)

        i_params_mtf = indicator_params_mtf[_i]
        b_params = backtest_params[_i]

        i_output_mtf = indicators_output_mtf[_i]
        s_output = signals_output[_i]
        b_output = backtest_output[_i]
        p_output = performance_output[_i]

        for m in range(mtf_count):
            _ohlcv = _ohlcv_mtf[m]
            _i_params = i_params_mtf[m]
            _i_output = i_output_mtf[m]
            calc_indicators(_ohlcv, _i_params, _i_output)

        calc_signal(
            _ohlcv_mtf, data_mapping, i_params_mtf, i_output_mtf, s_output, b_params
        )

        calc_backtest(_ohlcv_mtf, b_params, s_output, b_output)

        calc_performance(_ohlcv_mtf, b_params, b_output, p_output)

        if is_only_performance:
            clear_list_element_at_index(
                _i,
                indicators_output_mtf,
                signals_output,
                backtest_output,
            )
    if is_only_performance:
        (
            indicators_output_mtf,
            signals_output,
            backtest_output,
            _,
        ) = init_output_all(params_count, mtf_count, True)

    return (
        indicators_output_mtf,
        signals_output,
        backtest_output,
        performance_output,
    )
