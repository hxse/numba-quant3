import numpy as np
from numba import njit, prange
from numba.core import types
from numba.typed import Dict, List
import time

# 模拟 numba_config
numba_config = {
    "cache": True,
    "nb": {"float": types.float64, "bool": types.boolean, "int": types.int64},
}
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]
cache = numba_config["cache"]


# 数据初始化
def create_test_data(n=10000):
    tohlcv = Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
    tohlcv["open"] = np.random.rand(n)
    tohlcv["high"] = np.random.rand(n)
    tohlcv["low"] = np.random.rand(n)
    tohlcv["close"] = np.random.rand(n)
    tohlcv["atr"] = np.random.rand(n)

    backtest_params = Dict.empty(key_type=types.unicode_type, value_type=nb_float)
    backtest_params["close_for_reversal"] = 1.0
    backtest_params["pct_sl_enable"] = 1.0
    backtest_params["pct_tp_enable"] = 1.0
    backtest_params["pct_tsl_enable"] = 1.0
    backtest_params["pct_sl"] = 0.02
    backtest_params["pct_tp"] = 0.04
    backtest_params["pct_tsl"] = 0.01
    backtest_params["atr_sl_enable"] = 1.0
    backtest_params["atr_tp_enable"] = 1.0
    backtest_params["atr_tsl_enable"] = 1.0
    backtest_params["atr_sl_multiplier"] = 2.0
    backtest_params["atr_tp_multiplier"] = 4.0
    backtest_params["atr_tsl_multiplier"] = 1.5
    backtest_params["psar_enable"] = 1.0
    backtest_params["psar_af0"] = 0.02
    backtest_params["psar_af_step"] = 0.02
    backtest_params["psar_max_af"] = 0.2

    signal_output = Dict.empty(key_type=types.unicode_type, value_type=nb_bool[:])
    signal_output["enter_long"] = np.zeros(n, dtype=np.bool_)
    signal_output["exit_long"] = np.zeros(n, dtype=np.bool_)
    signal_output["enter_short"] = np.zeros(n, dtype=np.bool_)
    signal_output["exit_short"] = np.zeros(n, dtype=np.bool_)

    backtest_output = Dict.empty(key_type=types.unicode_type, value_type=nb_float[:])
    backtest_output["position"] = np.zeros(n, dtype=np.float64)
    backtest_output["entry_price"] = np.full(n, np.nan, dtype=np.float64)
    backtest_output["exit_price"] = np.full(n, np.nan, dtype=np.float64)
    backtest_output["pct_sl_arr"] = np.full(n, np.nan, dtype=np.float64)
    backtest_output["pct_tp_arr"] = np.full(n, np.nan, dtype=np.float64)
    backtest_output["pct_tsl_arr"] = np.full(n, np.nan, dtype=np.float64)
    backtest_output["atr_sl_arr"] = np.full(n, np.nan, dtype=np.float64)
    backtest_output["atr_tp_arr"] = np.full(n, np.nan, dtype=np.float64)
    backtest_output["atr_tsl_arr"] = np.full(n, np.nan, dtype=np.float64)
    backtest_output["psar_is_long_arr"] = np.full(n, 0.0, dtype=np.float64)
    backtest_output["psar_current_arr"] = np.full(n, np.nan, dtype=np.float64)
    backtest_output["psar_ep_arr"] = np.full(n, np.nan, dtype=np.float64)
    backtest_output["psar_af_arr"] = np.full(n, np.nan, dtype=np.float64)
    backtest_output["psar_reversal_arr"] = np.full(n, np.nan, dtype=np.float64)

    return tohlcv, backtest_params, signal_output, backtest_output


# 方案 2：字典标量打包为元组，数组直接传递
@njit(cache=cache)
def calculate_exit_triggers_v2(
    i,
    target_price,
    backtest_params_tuple,
    enter_long_signal,
    exit_long_signal,
    enter_short_signal,
    exit_short_signal,
    position,
    entry_price,
    exit_price,
    open_arr,
    high_arr,
    low_arr,
    close_arr,
    atr_arr,
    pct_sl_arr,
    pct_tp_arr,
    pct_tsl_arr,
    atr_sl_arr,
    atr_tp_arr,
    atr_tsl_arr,
    psar_is_long_arr,
    psar_current_arr,
    psar_ep_arr,
    psar_af_arr,
    psar_reversal_arr,
):
    last_i = i - 1
    (
        close_for_reversal,
        pct_sl_enable,
        pct_tp_enable,
        pct_tsl_enable,
        pct_sl,
        pct_tp,
        pct_tsl,
        atr_sl_enable,
        atr_tp_enable,
        atr_tsl_enable,
        atr_sl_multiplier,
        atr_tp_multiplier,
        atr_tsl_multiplier,
        psar_enable,
        psar_af0,
        psar_af_step,
        psar_max_af,
    ) = backtest_params_tuple
    if position[i] == 1.0:
        exit_check_price = close_arr[i] if close_for_reversal > 0 else low_arr[i]
        if pct_sl_enable > 0 and exit_check_price < target_price * (1 - pct_sl):
            exit_long_signal[i] = True
            enter_long_signal[i] = False
            exit_short_signal[i] = False
        pct_sl_arr[i] = target_price * (1 - pct_sl)
    elif position[i] == -1.0:
        exit_check_price = close_arr[i] if close_for_reversal > 0 else high_arr[i]
        if pct_sl_enable > 0 and exit_check_price > target_price * (1 + pct_sl):
            exit_short_signal[i] = True
            enter_short_signal[i] = False
            exit_long_signal[i] = False
        pct_sl_arr[i] = target_price * (1 + pct_sl)


@njit(cache=cache, parallel=True)
def parallel_backtest_v2(tohlcv, backtest_params, signal_output, backtest_output):
    n = len(tohlcv["close"])
    backtest_params_tuple = (
        backtest_params["close_for_reversal"],
        backtest_params["pct_sl_enable"],
        backtest_params["pct_tp_enable"],
        backtest_params["pct_tsl_enable"],
        backtest_params["pct_sl"],
        backtest_params["pct_tp"],
        backtest_params["pct_tsl"],
        backtest_params["atr_sl_enable"],
        backtest_params["atr_tp_enable"],
        backtest_params["atr_tsl_enable"],
        backtest_params["atr_sl_multiplier"],
        backtest_params["atr_tp_multiplier"],
        backtest_params["atr_tsl_multiplier"],
        backtest_params["psar_enable"],
        backtest_params["psar_af0"],
        backtest_params["psar_af_step"],
        backtest_params["psar_max_af"],
    )
    for i in prange(1, n):
        target_price = tohlcv["open"][i]
        calculate_exit_triggers_v2(
            i,
            target_price,
            backtest_params_tuple,
            signal_output["enter_long"],
            signal_output["exit_long"],
            signal_output["enter_short"],
            signal_output["exit_short"],
            backtest_output["position"],
            backtest_output["entry_price"],
            backtest_output["exit_price"],
            tohlcv["open"],
            tohlcv["high"],
            tohlcv["low"],
            tohlcv["close"],
            tohlcv["atr"],
            backtest_output["pct_sl_arr"],
            backtest_output["pct_tp_arr"],
            backtest_output["pct_tsl_arr"],
            backtest_output["atr_sl_arr"],
            backtest_output["atr_tp_arr"],
            backtest_output["atr_tsl_arr"],
            backtest_output["psar_is_long_arr"],
            backtest_output["psar_current_arr"],
            backtest_output["psar_ep_arr"],
            backtest_output["psar_af_arr"],
            backtest_output["psar_reversal_arr"],
        )


# 方案 4：backtest_params 转元组，signal_output/backtest_output/tohlcv 打包为元组
@njit(cache=cache)
def calculate_exit_triggers_v4(i, target_price, backtest_params_tuple, data_tuple):
    last_i = i - 1
    signal_output, backtest_output, tohlcv = data_tuple
    (
        close_for_reversal,
        pct_sl_enable,
        pct_tp_enable,
        pct_tsl_enable,
        pct_sl,
        pct_tp,
        pct_tsl,
        atr_sl_enable,
        atr_tp_enable,
        atr_tsl_enable,
        atr_sl_multiplier,
        atr_tp_multiplier,
        atr_tsl_multiplier,
        psar_enable,
        psar_af0,
        psar_af_step,
        psar_max_af,
    ) = backtest_params_tuple
    enter_long_signal = signal_output["enter_long"]
    exit_long_signal = signal_output["exit_long"]
    enter_short_signal = signal_output["enter_short"]
    exit_short_signal = signal_output["exit_short"]
    position = backtest_output["position"]
    entry_price = backtest_output["entry_price"]
    exit_price = backtest_output["exit_price"]
    open_arr = tohlcv["open"]
    high_arr = tohlcv["high"]
    low_arr = tohlcv["low"]
    close_arr = tohlcv["close"]
    atr_arr = tohlcv["atr"]
    pct_sl_arr = backtest_output["pct_sl_arr"]
    pct_tp_arr = backtest_output["pct_tp_arr"]
    pct_tsl_arr = backtest_output["pct_tsl_arr"]
    atr_sl_arr = backtest_output["atr_sl_arr"]
    atr_tp_arr = backtest_output["atr_tp_arr"]
    atr_tsl_arr = backtest_output["atr_tsl_arr"]
    psar_is_long_arr = backtest_output["psar_is_long_arr"]
    psar_current_arr = backtest_output["psar_current_arr"]
    psar_ep_arr = backtest_output["psar_ep_arr"]
    psar_af_arr = backtest_output["psar_af_arr"]
    psar_reversal_arr = backtest_output["psar_reversal_arr"]

    if position[i] == 1.0:
        exit_check_price = close_arr[i] if close_for_reversal > 0 else low_arr[i]
        if pct_sl_enable > 0 and exit_check_price < target_price * (1 - pct_sl):
            exit_long_signal[i] = True
            enter_long_signal[i] = False
            exit_short_signal[i] = False
        pct_sl_arr[i] = target_price * (1 - pct_sl)
    elif position[i] == -1.0:
        exit_check_price = close_arr[i] if close_for_reversal > 0 else high_arr[i]
        if pct_sl_enable > 0 and exit_check_price > target_price * (1 + pct_sl):
            exit_short_signal[i] = True
            enter_short_signal[i] = False
            exit_long_signal[i] = False
        pct_sl_arr[i] = target_price * (1 + pct_sl)


@njit(cache=cache, parallel=True)
def parallel_backtest_v4(tohlcv, backtest_params, signal_output, backtest_output):
    n = len(tohlcv["close"])
    backtest_params_tuple = (
        backtest_params["close_for_reversal"],
        backtest_params["pct_sl_enable"],
        backtest_params["pct_tp_enable"],
        backtest_params["pct_tsl_enable"],
        backtest_params["pct_sl"],
        backtest_params["pct_tp"],
        backtest_params["pct_tsl"],
        backtest_params["atr_sl_enable"],
        backtest_params["atr_tp_enable"],
        backtest_params["atr_tsl_enable"],
        backtest_params["atr_sl_multiplier"],
        backtest_params["atr_tp_multiplier"],
        backtest_params["atr_tsl_multiplier"],
        backtest_params["psar_enable"],
        backtest_params["psar_af0"],
        backtest_params["psar_af_step"],
        backtest_params["psar_max_af"],
    )
    data_tuple = (signal_output, backtest_output, tohlcv)
    for i in prange(1, n):
        target_price = tohlcv["open"][i]
        calculate_exit_triggers_v4(i, target_price, backtest_params_tuple, data_tuple)


# 方案 5：backtest_params 转元组，所有数组在 prange 前提取
@njit(cache=cache)
def calculate_exit_triggers_v5(
    i,
    target_price,
    backtest_params_tuple,
    enter_long_signal,
    exit_long_signal,
    enter_short_signal,
    exit_short_signal,
    position,
    entry_price,
    exit_price,
    open_arr,
    high_arr,
    low_arr,
    close_arr,
    atr_arr,
    pct_sl_arr,
    pct_tp_arr,
    pct_tsl_arr,
    atr_sl_arr,
    atr_tp_arr,
    atr_tsl_arr,
    psar_is_long_arr,
    psar_current_arr,
    psar_ep_arr,
    psar_af_arr,
    psar_reversal_arr,
):
    last_i = i - 1
    (
        close_for_reversal,
        pct_sl_enable,
        pct_tp_enable,
        pct_tsl_enable,
        pct_sl,
        pct_tp,
        pct_tsl,
        atr_sl_enable,
        atr_tp_enable,
        atr_tsl_enable,
        atr_sl_multiplier,
        atr_tp_multiplier,
        atr_tsl_multiplier,
        psar_enable,
        psar_af0,
        psar_af_step,
        psar_max_af,
    ) = backtest_params_tuple
    if position[i] == 1.0:
        exit_check_price = close_arr[i] if close_for_reversal > 0 else low_arr[i]
        if pct_sl_enable > 0 and exit_check_price < target_price * (1 - pct_sl):
            exit_long_signal[i] = True
            enter_long_signal[i] = False
            exit_short_signal[i] = False
        pct_sl_arr[i] = target_price * (1 - pct_sl)
    elif position[i] == -1.0:
        exit_check_price = close_arr[i] if close_for_reversal > 0 else high_arr[i]
        if pct_sl_enable > 0 and exit_check_price > target_price * (1 + pct_sl):
            exit_short_signal[i] = True
            enter_short_signal[i] = False
            exit_long_signal[i] = False
        pct_sl_arr[i] = target_price * (1 + pct_sl)


@njit(cache=cache, parallel=True)
def parallel_backtest_v5(tohlcv, backtest_params, signal_output, backtest_output):
    n = len(tohlcv["close"])
    backtest_params_tuple = (
        backtest_params["close_for_reversal"],
        backtest_params["pct_sl_enable"],
        backtest_params["pct_tp_enable"],
        backtest_params["pct_tsl_enable"],
        backtest_params["pct_sl"],
        backtest_params["pct_tp"],
        backtest_params["pct_tsl"],
        backtest_params["atr_sl_enable"],
        backtest_params["atr_tp_enable"],
        backtest_params["atr_tsl_enable"],
        backtest_params["atr_sl_multiplier"],
        backtest_params["atr_tp_multiplier"],
        backtest_params["atr_tsl_multiplier"],
        backtest_params["psar_enable"],
        backtest_params["psar_af0"],
        backtest_params["psar_af_step"],
        backtest_params["psar_max_af"],
    )
    enter_long_signal = signal_output["enter_long"]
    exit_long_signal = signal_output["exit_long"]
    enter_short_signal = signal_output["enter_short"]
    exit_short_signal = signal_output["exit_short"]
    position = backtest_output["position"]
    entry_price = backtest_output["entry_price"]
    exit_price = backtest_output["exit_price"]
    open_arr = tohlcv["open"]
    high_arr = tohlcv["high"]
    low_arr = tohlcv["low"]
    close_arr = tohlcv["close"]
    atr_arr = tohlcv["atr"]
    pct_sl_arr = backtest_output["pct_sl_arr"]
    pct_tp_arr = backtest_output["pct_tp_arr"]
    pct_tsl_arr = backtest_output["pct_tsl_arr"]
    atr_sl_arr = backtest_output["atr_sl_arr"]
    atr_tp_arr = backtest_output["atr_tp_arr"]
    atr_tsl_arr = backtest_output["atr_tsl_arr"]
    psar_is_long_arr = backtest_output["psar_is_long_arr"]
    psar_current_arr = backtest_output["psar_current_arr"]
    psar_ep_arr = backtest_output["psar_ep_arr"]
    psar_af_arr = backtest_output["psar_af_arr"]
    psar_reversal_arr = backtest_output["psar_reversal_arr"]

    for i in prange(1, n):
        target_price = open_arr[i]
        calculate_exit_triggers_v5(
            i,
            target_price,
            backtest_params_tuple,
            enter_long_signal,
            exit_long_signal,
            enter_short_signal,
            exit_short_signal,
            position,
            entry_price,
            exit_price,
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            atr_arr,
            pct_sl_arr,
            pct_tp_arr,
            pct_tsl_arr,
            atr_sl_arr,
            atr_tp_arr,
            atr_tsl_arr,
            psar_is_long_arr,
            psar_current_arr,
            psar_ep_arr,
            psar_af_arr,
            psar_reversal_arr,
        )


# 方案 6：在 prange 前将标量和数组打包为元组
@njit(cache=cache)
def calculate_exit_triggers_v6(scalar_tuple, backtest_params_tuple, array_tuple):
    i, target_price = scalar_tuple
    last_i = i - 1
    (
        close_for_reversal,
        pct_sl_enable,
        pct_tp_enable,
        pct_tsl_enable,
        pct_sl,
        pct_tp,
        pct_tsl,
        atr_sl_enable,
        atr_tp_enable,
        atr_tsl_enable,
        atr_sl_multiplier,
        atr_tp_multiplier,
        atr_tsl_multiplier,
        psar_enable,
        psar_af0,
        psar_af_step,
        psar_max_af,
    ) = backtest_params_tuple
    (
        enter_long_signal,
        exit_long_signal,
        enter_short_signal,
        exit_short_signal,
        position,
        entry_price,
        exit_price,
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        atr_arr,
        pct_sl_arr,
        pct_tp_arr,
        pct_tsl_arr,
        atr_sl_arr,
        atr_tp_arr,
        atr_tsl_arr,
        psar_is_long_arr,
        psar_current_arr,
        psar_ep_arr,
        psar_af_arr,
        psar_reversal_arr,
    ) = array_tuple

    if position[i] == 1.0:
        exit_check_price = close_arr[i] if close_for_reversal > 0 else low_arr[i]
        if pct_sl_enable > 0 and exit_check_price < target_price * (1 - pct_sl):
            exit_long_signal[i] = True
            enter_long_signal[i] = False
            exit_short_signal[i] = False
        pct_sl_arr[i] = target_price * (1 - pct_sl)
    elif position[i] == -1.0:
        exit_check_price = close_arr[i] if close_for_reversal > 0 else high_arr[i]
        if pct_sl_enable > 0 and exit_check_price > target_price * (1 + pct_sl):
            exit_short_signal[i] = True
            enter_short_signal[i] = False
            exit_long_signal[i] = False
        pct_sl_arr[i] = target_price * (1 + pct_sl)


@njit(cache=cache, parallel=True)
def parallel_backtest_v6(tohlcv, backtest_params, signal_output, backtest_output):
    n = len(tohlcv["close"])
    backtest_params_tuple = (
        backtest_params["close_for_reversal"],
        backtest_params["pct_sl_enable"],
        backtest_params["pct_tp_enable"],
        backtest_params["pct_tsl_enable"],
        backtest_params["pct_sl"],
        backtest_params["pct_tp"],
        backtest_params["pct_tsl"],
        backtest_params["atr_sl_enable"],
        backtest_params["atr_tp_enable"],
        backtest_params["atr_tsl_enable"],
        backtest_params["atr_sl_multiplier"],
        backtest_params["atr_tp_multiplier"],
        backtest_params["atr_tsl_multiplier"],
        backtest_params["psar_enable"],
        backtest_params["psar_af0"],
        backtest_params["psar_af_step"],
        backtest_params["psar_max_af"],
    )
    array_tuple = (
        signal_output["enter_long"],
        signal_output["exit_long"],
        signal_output["enter_short"],
        signal_output["exit_short"],
        backtest_output["position"],
        backtest_output["entry_price"],
        backtest_output["exit_price"],
        tohlcv["open"],
        tohlcv["high"],
        tohlcv["low"],
        tohlcv["close"],
        tohlcv["atr"],
        backtest_output["pct_sl_arr"],
        backtest_output["pct_tp_arr"],
        backtest_output["pct_tsl_arr"],
        backtest_output["atr_sl_arr"],
        backtest_output["atr_tp_arr"],
        backtest_output["atr_tsl_arr"],
        backtest_output["psar_is_long_arr"],
        backtest_output["psar_current_arr"],
        backtest_output["psar_ep_arr"],
        backtest_output["psar_af_arr"],
        backtest_output["psar_reversal_arr"],
    )
    for i in prange(1, n):
        target_price = tohlcv["open"][i]
        scalar_tuple = (i, target_price)
        calculate_exit_triggers_v6(scalar_tuple, backtest_params_tuple, array_tuple)


# 方案 7：所有标量和数组在 prange 前赋值，直接传递
@njit(cache=cache)
def calculate_exit_triggers_v7(
    i,
    target_price,
    close_for_reversal,
    pct_sl_enable,
    pct_tp_enable,
    pct_tsl_enable,
    pct_sl,
    pct_tp,
    pct_tsl,
    atr_sl_enable,
    atr_tp_enable,
    atr_tsl_enable,
    atr_sl_multiplier,
    atr_tp_multiplier,
    atr_tsl_multiplier,
    psar_enable,
    psar_af0,
    psar_af_step,
    psar_max_af,
    enter_long_signal,
    exit_long_signal,
    enter_short_signal,
    exit_short_signal,
    position,
    entry_price,
    exit_price,
    open_arr,
    high_arr,
    low_arr,
    close_arr,
    atr_arr,
    pct_sl_arr,
    pct_tp_arr,
    pct_tsl_arr,
    atr_sl_arr,
    atr_tp_arr,
    atr_tsl_arr,
    psar_is_long_arr,
    psar_current_arr,
    psar_ep_arr,
    psar_af_arr,
    psar_reversal_arr,
):
    last_i = i - 1
    if position[i] == 1.0:
        exit_check_price = close_arr[i] if close_for_reversal > 0 else low_arr[i]
        if pct_sl_enable > 0 and exit_check_price < target_price * (1 - pct_sl):
            exit_long_signal[i] = True
            enter_long_signal[i] = False
            exit_short_signal[i] = False
        pct_sl_arr[i] = target_price * (1 - pct_sl)
    elif position[i] == -1.0:
        exit_check_price = close_arr[i] if close_for_reversal > 0 else high_arr[i]
        if pct_sl_enable > 0 and exit_check_price > target_price * (1 + pct_sl):
            exit_short_signal[i] = True
            enter_short_signal[i] = False
            exit_long_signal[i] = False
        pct_sl_arr[i] = target_price * (1 + pct_sl)


@njit(cache=cache, parallel=True)
def parallel_backtest_v7(tohlcv, backtest_params, signal_output, backtest_output):
    n = len(tohlcv["close"])
    close_for_reversal = backtest_params["close_for_reversal"]
    pct_sl_enable = backtest_params["pct_sl_enable"]
    pct_tp_enable = backtest_params["pct_tp_enable"]
    pct_tsl_enable = backtest_params["pct_tsl_enable"]
    pct_sl = backtest_params["pct_sl"]
    pct_tp = backtest_params["pct_tp"]
    pct_tsl = backtest_params["pct_tsl"]
    atr_sl_enable = backtest_params["atr_sl_enable"]
    atr_tp_enable = backtest_params["atr_tp_enable"]
    atr_tsl_enable = backtest_params["atr_tsl_enable"]
    atr_sl_multiplier = backtest_params["atr_sl_multiplier"]
    atr_tp_multiplier = backtest_params["atr_tp_multiplier"]
    atr_tsl_multiplier = backtest_params["atr_tsl_multiplier"]
    psar_enable = backtest_params["psar_enable"]
    psar_af0 = backtest_params["psar_af0"]
    psar_af_step = backtest_params["psar_af_step"]
    psar_max_af = backtest_params["psar_max_af"]
    enter_long_signal = signal_output["enter_long"]
    exit_long_signal = signal_output["exit_long"]
    enter_short_signal = signal_output["enter_short"]
    exit_short_signal = signal_output["exit_short"]
    position = backtest_output["position"]
    entry_price = backtest_output["entry_price"]
    exit_price = backtest_output["exit_price"]
    open_arr = tohlcv["open"]
    high_arr = tohlcv["high"]
    low_arr = tohlcv["low"]
    close_arr = tohlcv["close"]
    atr_arr = tohlcv["atr"]
    pct_sl_arr = backtest_output["pct_sl_arr"]
    pct_tp_arr = backtest_output["pct_tp_arr"]
    pct_tsl_arr = backtest_output["pct_tsl_arr"]
    atr_sl_arr = backtest_output["atr_sl_arr"]
    atr_tp_arr = backtest_output["atr_tp_arr"]
    atr_tsl_arr = backtest_output["atr_tsl_arr"]
    psar_is_long_arr = backtest_output["psar_is_long_arr"]
    psar_current_arr = backtest_output["psar_current_arr"]
    psar_ep_arr = backtest_output["psar_ep_arr"]
    psar_af_arr = backtest_output["psar_af_arr"]
    psar_reversal_arr = backtest_output["psar_reversal_arr"]

    for i in prange(1, n):
        target_price = open_arr[i]
        calculate_exit_triggers_v7(
            i,
            target_price,
            close_for_reversal,
            pct_sl_enable,
            pct_tp_enable,
            pct_tsl_enable,
            pct_sl,
            pct_tp,
            pct_tsl,
            atr_sl_enable,
            atr_tp_enable,
            atr_tsl_enable,
            atr_sl_multiplier,
            atr_tp_multiplier,
            atr_tsl_multiplier,
            psar_enable,
            psar_af0,
            psar_af_step,
            psar_max_af,
            enter_long_signal,
            exit_long_signal,
            enter_short_signal,
            exit_short_signal,
            position,
            entry_price,
            exit_price,
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            atr_arr,
            pct_sl_arr,
            pct_tp_arr,
            pct_tsl_arr,
            atr_sl_arr,
            atr_tp_arr,
            atr_tsl_arr,
            psar_is_long_arr,
            psar_current_arr,
            psar_ep_arr,
            psar_af_arr,
            psar_reversal_arr,
        )


# 测试性能
def run_tests():
    n = 10000
    tohlcv, backtest_params, signal_output, backtest_output = create_test_data(n)
    backtest_output["position"][1:] = np.random.choice([0.0, 1.0, -1.0], size=n - 1)

    start = time.time()
    parallel_backtest_v2(tohlcv, backtest_params, signal_output, backtest_output)
    print(f"方案 2 (prange, 元组): {time.time() - start:.6f} 秒")

    tohlcv, backtest_params, signal_output, backtest_output = create_test_data(n)
    backtest_output["position"][1:] = np.random.choice([0.0, 1.0, -1.0], size=n - 1)

    start = time.time()
    parallel_backtest_v4(tohlcv, backtest_params, signal_output, backtest_output)
    print(f"方案 4 (prange, 字典打包元组): {time.time() - start:.6f} 秒")

    tohlcv, backtest_params, signal_output, backtest_output = create_test_data(n)
    backtest_output["position"][1:] = np.random.choice([0.0, 1.0, -1.0], size=n - 1)

    start = time.time()
    parallel_backtest_v5(tohlcv, backtest_params, signal_output, backtest_output)
    print(f"方案 5 (prange, 数组预提取): {time.time() - start:.6f} 秒")

    tohlcv, backtest_params, signal_output, backtest_output = create_test_data(n)
    backtest_output["position"][1:] = np.random.choice([0.0, 1.0, -1.0], size=n - 1)

    start = time.time()
    parallel_backtest_v6(tohlcv, backtest_params, signal_output, backtest_output)
    print(f"方案 6 (prange, 标量和数组打包元组): {time.time() - start:.6f} 秒")

    tohlcv, backtest_params, signal_output, backtest_output = create_test_data(n)
    backtest_output["position"][1:] = np.random.choice([0.0, 1.0, -1.0], size=n - 1)

    start = time.time()
    parallel_backtest_v7(tohlcv, backtest_params, signal_output, backtest_output)
    print(f"方案 7 (prange, 标量和数组直接传递): {time.time() - start:.6f} 秒")


if __name__ == "__main__":
    run_tests()
