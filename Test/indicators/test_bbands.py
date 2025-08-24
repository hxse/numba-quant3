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


name = "bbands"
# params=[[14, 2.0], [50, 2.5], [200, 3.0]]
params = [[14, 2.0]]

bbands_percent_rtol = 2e-4


def test_accuracy(
    np_data_mock,
    df_data_mock,
    talib=False,
    assert_mode=True,
):
    """
    bbands的实现和pandas-ta和talib, 都保持一致
    只有bbands_percent,需要调高容错到 2e-4
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
        (period, std_mult) = p

        target_array = np.full(params_count, 1.0, dtype=np_float)
        set_params_list_value(f"{name}_enable", indicator_params_list, target_array)

        target_array = np.full(params_count, period, dtype=np_float)
        set_params_list_value(f"{name}_period", indicator_params_list, target_array)

        target_array = np.full(params_count, std_mult, dtype=np_float)
        set_params_list_value(f"{name}_std_mult", indicator_params_list, target_array)

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

        np_upper = indicators_output_list[0][f"{name}_upper"]
        np_middle = indicators_output_list[0][f"{name}_middle"]
        np_lower = indicators_output_list[0][f"{name}_lower"]
        np_bandwidth = indicators_output_list[0][f"{name}_bandwidth"]
        np_percent = indicators_output_list[0][f"{name}_percent"]

        _func = getattr(ta, name)
        pandas_result = _func(
            close_series, length=int(period), std=std_mult, talib=talib
        )

        df_upper = pandas_result[f"BBU_{period}_{std_mult}"]
        df_middle = pandas_result[f"BBM_{period}_{std_mult}"]
        df_lower = pandas_result[f"BBL_{period}_{std_mult}"]
        df_bandwidth = pandas_result[f"BBB_{period}_{std_mult}"]
        df_percent = pandas_result[f"BBP_{period}_{std_mult}"]

        for i in [
            [np_upper, df_upper, f"{name}_upper"],
            [np_middle, df_middle, f"{name}_middle"],
            [np_lower, df_lower, f"{name}_lower"],
            [np_bandwidth, df_bandwidth, f"{name}_bandwidth"],
            [np_percent, df_percent, f"{name}_percent"],
        ]:
            np_array, df_array, _name = i

            custom_params = {}
            if _name == f"{name}_percent":
                custom_params["custom_rtol"] = bbands_percent_rtol

            assert_func = (
                assert_indicator_same if assert_mode else assert_indicator_different
            )
            assert_func(
                np_array,
                df_array,
                _name,
                f"period {period} std_mult {std_mult}",
                **custom_params,
            )


def test_accuracy_talib(np_data_mock, df_data_mock, talib=True, assert_mode=True):
    test_accuracy(np_data_mock, df_data_mock, talib=talib, assert_mode=assert_mode)


def test_pandas_ta_and_talib(df_data_mock, assert_mode=True):
    """
    对于bbands, pandas-ta和talib的两种实现版本, 预期一致
    只有bbands_percent,需要调高容错到 2e-4
    """
    time_series = df_data_mock["time"]
    open_series = df_data_mock["open"]
    high_series = df_data_mock["high"]
    low_series = df_data_mock["low"]
    close_series = df_data_mock["close"]
    volume_series = df_data_mock["volume"]

    for p in params:
        (period, std_mult) = p

        _func = getattr(ta, name)

        # 使用 pandas_ta 计算指标

        _func = getattr(ta, name)
        pandas_ta_result = _func(
            close_series, length=int(period), std=std_mult, talib=False
        )

        pandas_ta_upper = pandas_ta_result[f"BBU_{period}_{std_mult}"]
        pandas_ta_middle = pandas_ta_result[f"BBM_{period}_{std_mult}"]
        pandas_ta_lower = pandas_ta_result[f"BBL_{period}_{std_mult}"]
        pandas_ta_bandwidth = pandas_ta_result[f"BBB_{period}_{std_mult}"]
        pandas_ta_percent = pandas_ta_result[f"BBP_{period}_{std_mult}"]

        # 使用 talib 计算指标
        talib_result = _func(close_series, length=int(period), std=std_mult, talib=True)

        talib_upper = talib_result[f"BBU_{period}_{std_mult}"]
        talib_middle = talib_result[f"BBM_{period}_{std_mult}"]
        talib_lower = talib_result[f"BBL_{period}_{std_mult}"]
        talib_bandwidth = talib_result[f"BBB_{period}_{std_mult}"]
        talib_percent = talib_result[f"BBP_{period}_{std_mult}"]

        for i in [
            [pandas_ta_upper, talib_upper, f"{name}_upper"],
            [pandas_ta_middle, talib_middle, f"{name}_middle"],
            [pandas_ta_lower, talib_lower, f"{name}_lower"],
            [pandas_ta_bandwidth, talib_bandwidth, f"{name}_bandwidth"],
            [pandas_ta_percent, talib_percent, f"{name}_percent"],
        ]:
            pandas_ta_array, talib_array, _name = i

            custom_params = {}
            if _name == f"{name}_percent":
                custom_params["custom_rtol"] = bbands_percent_rtol

            assert_func = (
                assert_indicator_same if assert_mode else assert_indicator_different
            )
            assert_func(
                pandas_ta_array, talib_array, _name, f"period {period}", **custom_params
            )
