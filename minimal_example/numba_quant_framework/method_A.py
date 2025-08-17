import numpy as np
from numba import njit, prange
from numba.core import types
from numba.typed import Dict, List


np_int = np.int64
np_float = np.float64
np_bool = np.bool
nb_int = types.int64
nb_float = types.float64
nb_bool = types.bool

np.random.seed(42)

data_count = 5
tohlcv = Dict.empty(
    key_type=types.unicode_type,
    value_type=nb_float[:],
)
tohlcv["close"] = np.random.rand(data_count).astype(np.float64)


@njit
def get_indicator_params(n):
    params = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float,
    )
    params["sma_period"] = 14 + n
    params["sma2_period"] = 14 + n + 100
    params["bbands_period"] = 20 + n
    params["bbands_std_mult"] = 2.0
    return params


@njit
def get_backtest_params(n):
    params = Dict.empty(
        key_type=types.unicode_type,
        value_type=nb_float,
    )
    params["signal_select"] = 0
    params["atr_sl_mult"] = 3
    return params


@njit
def calc_sma(close, sma_period):
    num_data = len(close)
    res_sma = np.empty(num_data, dtype=np_float)
    res_sma = close + sma_period
    return res_sma


@njit
def calc_bbands(close, bbands_period, bbands_std_mult):
    num_data = len(close)
    res_bbands = np.empty((num_data, 3), dtype=np_float)
    res_bbands[:, 0] = close + bbands_period + 1 * bbands_std_mult
    res_bbands[:, 1] = close + bbands_period + 2 * bbands_std_mult
    res_bbands[:, 2] = close + bbands_period + 3 * bbands_std_mult

    return res_bbands


@njit
def calc_indicators(indicator_item, close, param):
    indicator_item["sma"] = calc_sma(close, param["sma_period"])
    indicator_item["sma2"] = calc_sma(close, param["sma2_period"])
    bbands = calc_bbands(close, param["bbands_period"], param["bbands_std_mult"])
    indicator_item["bbands_upper"] = bbands[:, 0]
    indicator_item["bbands_middle"] = bbands[:, 1]
    indicator_item["bbands_lower"] = bbands[:, 2]


@njit
def calc_signal_0(signal_item, indicator_item, close):
    sma = indicator_item["sma"]
    sma2 = indicator_item["sma2"]
    signal_item["enter_long"] = sma > sma2
    signal_item["exit_long"] = sma < sma2
    signal_item["enter_short"] = sma < sma2
    signal_item["exit_short"] = sma > sma2


@njit
def calc_signal_1(signal_item, indicator_item, close):
    bbands_upper = indicator_item["bbands_upper"]
    bbands_middle = indicator_item["bbands_middle"]
    bbands_lower = indicator_item["bbands_lower"]
    signal_item["enter_long"] = close < bbands_lower
    signal_item["exit_long"] = close > bbands_middle
    signal_item["enter_short"] = close > bbands_upper
    signal_item["exit_short"] = close < bbands_middle


@njit
def calc_signal(signal_item, indicator_item, close, backtest_params):
    # 使用 if/elif 结构进行分派
    if backtest_params["signal_select"] == 0.0:
        calc_signal_0(signal_item, indicator_item, close)
    elif backtest_params["signal_select"] == 1.0:
        calc_signal_1(signal_item, indicator_item, close)


@njit
def calc_backtest(backtest_item, signal_item, close, backtest_params):
    num_data = len(close)
    position = np.empty(num_data, dtype=np_float)
    price = np.empty(num_data, dtype=np_float)
    money = np.empty(num_data, dtype=np_float)
    enter_long = signal_item["enter_long"]
    exit_long = signal_item["exit_long"]

    for i in range(len(close)):
        money[i] = close[i]
        if enter_long[i]:
            position[i] = 1
            price[i] = close[i]
        elif exit_long[i]:
            position[i] = 0
            price[i] = close[i]

    backtest_item["position"] = position
    backtest_item["price"] = price
    backtest_item["money"] = money


@njit
def calc_performance(performance_item, backtest_item, close, backtest_params):
    # 这里实现最大回撤、夏普比率等计算
    position = backtest_item["position"]
    price = backtest_item["price"]
    money = backtest_item["money"]

    max_money = 0
    max_drawdown = 0
    for i in range(len(money)):
        max_money = max(money[i], max_money)
        max_drawdown = max(max_money - money[i], max_drawdown)

    performance_item["max_money"] = max_money
    performance_item["max_drawdown"] = max_drawdown


import numpy as np
from numba.core import types
from numba.typed import Dict, List


@njit
def create_backtest_containers(count):
    # 创建 Typed Lists
    indicators_list = List()
    signals_list = List()
    backtest_list = List()
    performance_list = List()

    for _ in range(count):
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

    return indicators_list, signals_list, backtest_list, performance_list


@njit(parallel=True)
def entry(tohlcv, indicator_params_list, backtest_params_list):
    assert len(indicator_params_list) == len(backtest_params_list), (
        "参数组合数量需要相等"
    )

    # 调用工具函数，一行代码完成预分配
    indicators_list, signals_list, backtest_list, performance_list = (
        create_backtest_containers(len(indicator_params_list))
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


if __name__ == "__main__":
    params_count = 3
    indicators_params_list = List(
        get_indicator_params((n + 1) * 10) for n in range(params_count)
    )
    backtest_params_list = List(
        get_backtest_params((n + 1) * 10) for n in range(params_count)
    )

    # 调用函数，传入所有必要的参数
    res = entry(tohlcv, indicators_params_list, backtest_params_list)
    print("indicators_list", res[0])
    print("signals_list", res[1])
    print("backtest_list", res[2])
    print("performance_list", res[3])
