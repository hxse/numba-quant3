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

# 模拟 tohlcv 和 backtest_params 字典类型
tohlcv_np_type = types.DictType(types.unicode_type, nb_float[:])
param_dict_type = types.DictType(types.unicode_type, nb_float)

# 模拟信号和回测输出字典
signal_output_type = types.DictType(types.unicode_type, nb_bool[:])
backtest_output_type = types.DictType(types.unicode_type, nb_float[:])


# 初始化数据
def create_test_data(n=1000):
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


# 方案 1：直接传递字典和标量
@njit(cache=cache)
def calculate_exit_triggers_v1(
    i,
    target_price,
    backtest_params,
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
    # 模拟简单的止损逻辑
    if position[i] == 1.0:  # 假设 1.0 表示多头
        exit_check_price = (
            close_arr[i] if backtest_params["close_for_reversal"] > 0 else low_arr[i]
        )
        if backtest_params["pct_sl_enable"] > 0 and exit_check_price < target_price * (
            1 - backtest_params["pct_sl"]
        ):
            exit_long_signal[i] = True
            enter_long_signal[i] = False
            exit_short_signal[i] = False
        pct_sl_arr[i] = target_price * (1 - backtest_params["pct_sl"])
    elif position[i] == -1.0:  # 假设 -1.0 表示空头
        exit_check_price = (
            close_arr[i] if backtest_params["close_for_reversal"] > 0 else high_arr[i]
        )
        if backtest_params["pct_sl_enable"] > 0 and exit_check_price > target_price * (
            1 + backtest_params["pct_sl"]
        ):
            exit_short_signal[i] = True
            enter_short_signal[i] = False
            exit_long_signal[i] = False
        pct_sl_arr[i] = target_price * (1 + backtest_params["pct_sl"])


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


# 方案 3：所有标量打包为元组，数组直接传递
@njit(cache=cache)
def calculate_exit_triggers_v3(
    i,
    params_tuple,
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
        target_price,
        close_for_reversal,
        pct_sl_enable,
        pct_tp_enable,
        pct_tsl_enable,
        pct_sl,
        pct_tp,
        pct_tsl,
        atr_sl_multiplier,
        atr_tp_multiplier,
        atr_tsl_multiplier,
        psar_enable,
        psar_af0,
        psar_af_step,
        psar_max_af,
    ) = params_tuple
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


# 主回测函数，模拟 range 循环
@njit(cache=cache)
def backtest_v1(tohlcv, backtest_params, signal_output, backtest_output):
    n = len(tohlcv["close"])
    for i in range(1, n):
        target_price = tohlcv["open"][i]
        calculate_exit_triggers_v1(
            i,
            target_price,
            backtest_params,
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


@njit(cache=cache)
def backtest_v2(tohlcv, backtest_params, signal_output, backtest_output):
    n = len(tohlcv["close"])
    backtest_params_tuple = (
        backtest_params["close_for_reversal"],
        backtest_params["pct_sl_enable"],
        backtest_params["pct_tp_enable"],
        backtest_params["pct_tsl_enable"],
        backtest_params["pct_sl"],
        backtest_params["pct_tp"],
        backtest_params["pct_tsl"],
        backtest_params["atr_sl_multiplier"],
        backtest_params["atr_tp_multiplier"],
        backtest_params["atr_tsl_multiplier"],
        backtest_params["psar_enable"],
        backtest_params["psar_af0"],
        backtest_params["psar_af_step"],
        backtest_params["psar_max_af"],
    )
    for i in range(1, n):
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


@njit(cache=cache)
def backtest_v3(tohlcv, backtest_params, signal_output, backtest_output):
    n = len(tohlcv["close"])
    backtest_params_tuple = (
        tohlcv["open"][1],  # target_price
        backtest_params["close_for_reversal"],
        backtest_params["pct_sl_enable"],
        backtest_params["pct_tp_enable"],
        backtest_params["pct_tsl_enable"],
        backtest_params["pct_sl"],
        backtest_params["pct_tp"],
        backtest_params["pct_tsl"],
        backtest_params["atr_sl_multiplier"],
        backtest_params["atr_tp_multiplier"],
        backtest_params["atr_tsl_multiplier"],
        backtest_params["psar_enable"],
        backtest_params["psar_af0"],
        backtest_params["psar_af_step"],
        backtest_params["psar_max_af"],
    )
    for i in range(1, n):
        calculate_exit_triggers_v3(
            i,
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


# 模拟 prange 的并行调用
@njit(cache=cache, parallel=True)
def parallel_backtest_v1(tohlcv, backtest_params, signal_output, backtest_output):
    n = len(tohlcv["close"])
    for i in prange(1, n):
        target_price = tohlcv["open"][i]
        calculate_exit_triggers_v1(
            i,
            target_price,
            backtest_params,
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


@njit(cache=cache, parallel=True)
def parallel_backtest_v3(tohlcv, backtest_params, signal_output, backtest_output):
    n = len(tohlcv["close"])
    for i in prange(1, n):
        params_tuple = (
            tohlcv["open"][i],  # target_price
            backtest_params["close_for_reversal"],
            backtest_params["pct_sl_enable"],
            backtest_params["pct_tp_enable"],
            backtest_params["pct_tsl_enable"],
            backtest_params["pct_sl"],
            backtest_params["pct_tp"],
            backtest_params["pct_tsl"],
            backtest_params["atr_sl_multiplier"],
            backtest_params["atr_tp_multiplier"],
            backtest_params["atr_tsl_multiplier"],
            backtest_params["psar_enable"],
            backtest_params["psar_af0"],
            backtest_params["psar_af_step"],
            backtest_params["psar_max_af"],
        )
        calculate_exit_triggers_v3(
            i,
            params_tuple,
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


# 测试性能
def run_tests():
    n = 10000
    tohlcv, backtest_params, signal_output, backtest_output = create_test_data(n)
    # 模拟持仓状态
    backtest_output["position"][1:] = np.random.choice([0.0, 1.0, -1.0], size=n - 1)

    # 测试方案 1
    start = time.time()
    backtest_v1(tohlcv, backtest_params, signal_output, backtest_output)
    print(f"方案 1 (range, 字典): {time.time() - start:.6f} 秒")
    parallel_backtest_v1(tohlcv, backtest_params, signal_output, backtest_output)
    print(f"方案 1 (prange, 字典): {time.time() - start:.6f} 秒")

    # 重置数据
    tohlcv, backtest_params, signal_output, backtest_output = create_test_data(n)
    backtest_output["position"][1:] = np.random.choice([0.0, 1.0, -1.0], size=n - 1)

    # 测试方案 2
    start = time.time()
    backtest_v2(tohlcv, backtest_params, signal_output, backtest_output)
    print(f"方案 2 (range, 元组): {time.time() - start:.6f} 秒")
    parallel_backtest_v2(tohlcv, backtest_params, signal_output, backtest_output)
    print(f"方案 2 (prange, 元组): {time.time() - start:.6f} 秒")

    # 重置数据
    tohlcv, backtest_params, signal_output, backtest_output = create_test_data(n)
    backtest_output["position"][1:] = np.random.choice([0.0, 1.0, -1.0], size=n - 1)

    # 测试方案 3
    start = time.time()
    backtest_v3(tohlcv, backtest_params, signal_output, backtest_output)
    print(f"方案 3 (range, 所有标量元组): {time.time() - start:.6f} 秒")
    parallel_backtest_v3(tohlcv, backtest_params, signal_output, backtest_output)
    print(f"方案 3 (prange, 所有标量元组): {time.time() - start:.6f} 秒")


if __name__ == "__main__":
    run_tests()
