# calc_backtest.py
import numpy as np
from numba import njit
from numba.core import types
from numba.typed import List, Dict

from src.utils.constants import numba_config
from src.utils.nb_check_keys import check_keys, check_tohlcv_keys

from src.backtest.calculate_trade_logic import calc_trade_logic
from src.backtest.calculate_balance import calc_balance
from backtest.calculate_exit_logic import calc_exit_logic

from src.indicators.atr import calc_atr

from src.parallel_signature import backtest_signature


enable_cache = numba_config["enable_cache"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


@njit(cache=enable_cache)
def get_backtest_keys():
    _l = List.empty_list(types.unicode_type)
    for i in ("enter_long", "exit_long", "enter_short", "exit_short"):
        _l.append(i)
    return _l


@njit(cache=enable_cache)
def get_backtest_params_keys():
    _l = List.empty_list(types.unicode_type)
    for i in (
        "init_money",
        "close_for_reversal",
        "pct_sl",
        "pct_tp",
        "pct_tsl",
        "pct_sl_enable",
        "pct_tp_enable",
        "pct_tsl_enable",
        "atr_period",
        "atr_sl_multiplier",
        "atr_tp_multiplier",
        "atr_tsl_multiplier",
        "atr_sl_enable",
        "atr_tp_enable",
        "atr_tsl_enable",
        "psar_enable",
        "psar_af0",
        "psar_af_step",
        "psar_max_af",
        "commission_pct",
        "commission_fixed",
        "slippage_atr",
        "slippage_pct",
        "position_size",
    ):
        _l.append(i)
    return _l


@njit(backtest_signature, cache=enable_cache)
def calc_backtest(tohlcv, backtest_params, signal_output, backtest_output):
    """
    backtest_output["position"] 代表仓位状态,0无仓位,1开多,2持多,3平多,4平空开多,-1开空,-2持空,-3平空,-4平多开空
    Bar-by-Bar模式,在触发信号的下一根k线的开盘价离场,为了简化不考虑k线内部实时离场的功能
    比如无论索引last_i是触发止盈,还是触发止损,还是同时触发止盈止损,都会在索引i的open价格离场,没有区别,这样设计是为了简化回测
    可以多头,可以空头,但是每次只持一仓
    """
    # 1. 输入数据校验
    if not check_keys(get_backtest_keys(), signal_output):
        return

    if not check_tohlcv_keys(tohlcv):
        return

    if not check_keys(get_backtest_params_keys(), backtest_params):
        return

    # 2. 从字典中提取数据数组
    open_arr = tohlcv["open"]
    high_arr = tohlcv["high"]
    low_arr = tohlcv["low"]
    close_arr = tohlcv["close"]
    volume_arr = tohlcv["volume"]

    data_count = len(close_arr)

    enter_long_signal = signal_output["enter_long"]
    exit_long_signal = signal_output["exit_long"]
    enter_short_signal = signal_output["enter_short"]
    exit_short_signal = signal_output["exit_short"]

    # 3. 初始化回测结果数组
    position = np.full(data_count, 0, dtype=nb_float)
    entry_price = np.full(data_count, np.nan, dtype=nb_float)
    exit_price = np.full(data_count, np.nan, dtype=nb_float)
    equity = np.full(data_count, np.nan, dtype=nb_float)
    balance = np.full(data_count, np.nan, dtype=nb_float)
    drawdown = np.full(data_count, np.nan, dtype=nb_float)

    # 4. 初始化临时变量和止损参数
    init_money = backtest_params["init_money"]
    max_balance = np.full(data_count, np.nan, dtype=nb_float)

    # 计算 ATR 数组
    atr_arr = calc_atr(high_arr, low_arr, close_arr, backtest_params["atr_period"])

    # 临时数组用于存储止损价格和 PSAR 状态
    pct_sl_arr = np.full(data_count, np.nan, dtype=nb_float)
    pct_tp_arr = np.full(data_count, np.nan, dtype=nb_float)
    pct_tsl_arr = np.full(data_count, np.nan, dtype=nb_float)
    atr_sl_arr = np.full(data_count, np.nan, dtype=nb_float)
    atr_tp_arr = np.full(data_count, np.nan, dtype=nb_float)
    atr_tsl_arr = np.full(data_count, np.nan, dtype=nb_float)
    psar_is_long_arr = np.full(data_count, 0.0, dtype=nb_float)
    psar_current_arr = np.full(data_count, np.nan, dtype=nb_float)
    psar_ep_arr = np.full(data_count, np.nan, dtype=nb_float)
    psar_af_arr = np.full(data_count, np.nan, dtype=nb_float)
    psar_reversal_arr = np.full(data_count, np.nan, dtype=nb_float)

    # 初始化第一天数据
    balance[0] = init_money
    equity[0] = init_money
    max_balance[0] = init_money
    drawdown[0] = 0.0

    # numba传参有奇怪的优化问题,这里必须打包成元组,提高性能
    backtest_params_tuple = (
        backtest_params["close_for_reversal"],  # 用close触发止损,还是high和low
        backtest_params["pct_sl_enable"],  # 是否开启百分比止损
        backtest_params["pct_tp_enable"],  # 是否开启百分比止盈
        backtest_params["pct_tsl_enable"],  # 是否开启百分比跟踪止损
        backtest_params["pct_sl"],  # 百分比止损倍率，例如0.02代表2%
        backtest_params["pct_tp"],  # 百分比止盈倍率，例如0.05代表5%
        backtest_params["pct_tsl"],  # 百分比跟踪止损倍率
        backtest_params["atr_sl_enable"],  # 是否开启ATR止损
        backtest_params["atr_tp_enable"],  # 是否开启ATR止盈
        backtest_params["atr_tsl_enable"],  # 是否开启ATR跟踪止损
        backtest_params["atr_sl_multiplier"],  # ATR止损倍数，例如3代表3倍ATR
        backtest_params["atr_tp_multiplier"],  # ATR止盈倍数
        backtest_params["atr_tsl_multiplier"],  # ATR跟踪止损倍数
        backtest_params["psar_enable"],  # 是否开启PSAR止损
        backtest_params["psar_af0"],  # PSAR的加速因子初始值
        backtest_params["psar_af_step"],  # PSAR的加速因子步长
        backtest_params["psar_max_af"],  # PSAR的最大加速因子
    )

    commission_pct = backtest_params["commission_pct"]  # 基于百分比的手续费
    commission_fixed = backtest_params["commission_fixed"]  # 固定金额的手续费
    slippage_atr = backtest_params["slippage_atr"]  # 基于ATR的滑点倍数，用于计算滑点
    slippage_pct = backtest_params["slippage_pct"]  # 基于百分比的滑点，用于计算滑点
    # 仓位大小：如果为0-1之间的小数，表示资金百分比；如果为大于等于1的整数(类型依然是小数)，则表示杠杆倍数
    position_size = backtest_params["position_size"]

    # 5. 主循环
    for i in range(1, data_count):
        last_i = i - 1
        target_price = open_arr[i]

        # 仓位和进出场价格逻辑
        calc_trade_logic(
            i,
            enter_long_signal,
            exit_long_signal,
            enter_short_signal,
            exit_short_signal,
            position,
            entry_price,
            exit_price,
            target_price,
        )

        # 计算止损止盈触发（可能修改 signal 数组）
        calc_exit_logic(
            i,
            target_price,
            backtest_params_tuple,
            #
            enter_long_signal,
            exit_long_signal,
            enter_short_signal,
            exit_short_signal,
            #
            position,
            entry_price,
            exit_price,
            #
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            #
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

        # 资金、净值、回撤计算
        calc_balance(
            i,
            last_i,
            open_arr,
            close_arr,
            position,
            entry_price,
            exit_price,
            equity,
            balance,
            drawdown,
            max_balance,
            atr_arr,
            commission_pct,
            commission_fixed,
            slippage_atr,
            slippage_pct,
            position_size,
        )

    # 将结果保存到 backtest_output 字典
    backtest_output["position"] = position
    backtest_output["entry_price"] = entry_price
    backtest_output["exit_price"] = exit_price
    backtest_output["equity"] = equity
    backtest_output["balance"] = balance
    backtest_output["drawdown"] = drawdown

    # 将新的止损止盈和PSAR数组保存到 backtest_output
    backtest_output["pct_sl_arr"] = pct_sl_arr
    backtest_output["pct_tp_arr"] = pct_tp_arr
    backtest_output["pct_tsl_arr"] = pct_tsl_arr
    backtest_output["atr_sl_arr"] = atr_sl_arr
    backtest_output["atr_tp_arr"] = atr_tp_arr
    backtest_output["atr_tsl_arr"] = atr_tsl_arr
    backtest_output["psar_is_long_arr"] = psar_is_long_arr
    backtest_output["psar_current_arr"] = psar_current_arr
    backtest_output["psar_ep_arr"] = psar_ep_arr
    backtest_output["psar_af_arr"] = psar_af_arr
    backtest_output["psar_reversal_arr"] = psar_reversal_arr
