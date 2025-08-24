import sys
from pathlib import Path


root_path = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()),
    None,
)
if root_path:
    sys.path.insert(0, str(root_path))


from Test.utils.over_constants import numba_config


from Test.utils.comparison_tool import assert_indicator_different, assert_indicator_same


import numpy as np
import pandas_ta as ta

from src.utils.nb_params import (
    create_params_list_template,
    create_params_dict_template,
    get_params_list_value,
    set_params_list_value,
    get_params_dict_value,
    set_params_dict_value,
    convert_params_dict_list,
)
from src.parallel import run_parallel
from src.utils.handle_params import init_params
from src.signals.calculate_signal import SignalId, signal_dict
from Test.utils.conftest import np_data_mock, df_data_mock


np_float = numba_config["np"]["float"]


name = "rsi"
# params = [[14], [50], [200]]
params = [[14]]


def test_accuracy(
    np_data_mock,
    df_data_mock,
    talib=False,
    assert_mode=False,
):
    """
    rsi的实现和talib保持一致,和pandas-ta不同
    """
    time = np_data_mock[:, 0]
    open = np_data_mock[:, 1]
    high = np_data_mock[:, 2]
    low = np_data_mock[:, 3]
    close = np_data_mock[:, 4]
    volume = np_data_mock[:, 5]

    time_series = df_data_mock["time"]
    open_series = df_data_mock["open"]
    high_series = df_data_mock["high"]
    low_series = df_data_mock["low"]
    close_series = df_data_mock["close"]
    volume_series = df_data_mock["volume"]

    tohlcv_np = np_data_mock

    params_count = 1
    signal_select_id = SignalId.signal_0_id.value

    (
        tohlcv_np,
        indicator_params_list,
        backtest_params_list,
        tohlcv_np_mtf,
        indicator_params_list_mtf,
        mapping_mtf,
        tohlcv_smoothed,
        tohlcv_mtf_smoothed,
    ) = init_params(
        params_count,
        signal_select_id,
        signal_dict,
        tohlcv_np,
        tohlcv_np_mtf=None,
        mapping_mtf=None,
        smooth_mode=None,
    )

    for p in params:
        (period,) = p

        target_array = np.full(params_count, 1.0, dtype=np_float)
        set_params_list_value(f"{name}_enable", indicator_params_list, target_array)

        target_array = np.full(params_count, period, dtype=np_float)
        set_params_list_value(f"{name}_period", indicator_params_list, target_array)

        result = run_parallel(
            tohlcv_np,
            indicator_params_list,
            backtest_params_list,
            tohlcv_np_mtf,
            indicator_params_list_mtf,
            mapping_mtf,
            tohlcv_smoothed,
            tohlcv_mtf_smoothed,
        )
        (
            indicators_output_list,
            signals_output_list,
            backtest_output_list,
            performance_output_list,
            indicators_output_list_mtf,
        ) = result

        np_result = indicators_output_list[0][name]

        _func = getattr(ta, name)
        pandas_result = _func(close_series, length=int(period), talib=talib)

        custom_params = {}

        assert_func = (
            assert_indicator_same if assert_mode else assert_indicator_different
        )
        assert_func(
            np_result,
            pandas_result,
            name,
            f"period {period}",
            **custom_params,
        )


def test_accuracy_talib(np_data_mock, df_data_mock, talib=True, assert_mode=True):
    test_accuracy(np_data_mock, df_data_mock, talib=talib, assert_mode=assert_mode)


def test_pandas_ta_and_talib(df_data_mock, assert_mode=False):
    """
    对于rsi, pandas-ta和talib的两种实现版本, 预期不一致
    """
    time_series = df_data_mock["time"]
    open_series = df_data_mock["open"]
    high_series = df_data_mock["high"]
    low_series = df_data_mock["low"]
    close_series = df_data_mock["close"]
    volume_series = df_data_mock["volume"]

    for p in params:
        (period,) = p

        _func = getattr(ta, name)

        # 使用 pandas_ta 计算指标
        pandas_result = _func(close_series, length=int(period), talib=False)

        # 使用 talib 计算指标
        talib_result = _func(close_series, length=int(period), talib=True)

        assert_func = (
            assert_indicator_same if assert_mode else assert_indicator_different
        )
        assert_func(pandas_result, talib_result, name, f"period {period}")
