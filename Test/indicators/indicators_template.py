import numpy as np
import pandas as pd
import pandas_ta as ta

from Test.utils.over_constants import numba_config

from src.utils.nb_params import set_params_list_value
from src.parallel import run_parallel
from src.utils.handle_params import init_params
from src.signals.calculate_signal import SignalId, signal_dict
from Test.utils.comparison_tool import assert_indicator_different, assert_indicator_same

np_float = numba_config["np"]["float"]


def compare_indicator_accuracy(
    name,
    params_config_list,
    tohlcv_np,
    df_data_mock,
    input_data_keys,
    talib=False,
    assert_mode=False,
    params_count=1,
    signal_select_id=SignalId.signal_0_id.value,
    assert_func_kwargs={},
    output_key_maps=None,
):
    """
    通用指标精度测试工具函数。
    name: 指标名称 (e.g., "atr", "sma", "bbands")
    params_config_list: 指标参数配置列表，每个元素是一个字典，包含nb_params和pd_params
                        e.g., [{"nb_params": {"period": 14}, "pd_params": {"length": 14}}]
    tohlcv_np: numpy格式的OHLCV数据
    df_data_mock: pandas DataFrame格式的OHLCV数据
    input_data_keys: 指标所需的输入数据键列表 (e.g., ["high", "low", "close"])
    talib: 是否使用pandas-ta的talib实现
    assert_mode: 断言模式（True为相等，False为不相等）
    output_key_maps: 可选参数，列表格式，每个元素是一个字典，包含Numba输出与pandas-ta输出的列名映射。
                     e.g., [{"nb_col1": "pd_col1", "nb_col2": "pd_col2"}]
    """
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

    # 统一处理单列和多列的情况
    if not output_key_maps:
        # 如果是单列，创建默认的 output_key_map 列表
        output_key_maps = [{name: name} for _ in params_config_list]

    assert len(params_config_list) == len(output_key_maps), (
        "params_config_list and output_key_maps must have the same length."
    )

    for i, params_config in enumerate(params_config_list):
        nb_params_dict = params_config.get("nb_params", {})
        pd_params_dict = params_config.get("pd_params", {})
        output_key_map = output_key_maps[i]

        # 动态设置Numba函数的参数
        for param_name, param_value in nb_params_dict.items():
            set_params_list_value(
                f"{name}_{param_name}",
                indicator_params_list,
                np.full(params_count, param_value, dtype=np_float),
            )

        set_params_list_value(
            f"{name}_enable",
            indicator_params_list,
            np.full(params_count, 1.0, dtype=np_float),
        )

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
            _,
            _,
            _,
            _,
        ) = result

        # 动态构建pandas-ta的函数调用参数
        pandas_ta_args = {**pd_params_dict, "talib": talib}
        input_series = [df_data_mock[key] for key in input_data_keys]

        _func = getattr(ta, name)
        pandas_result = _func(*input_series, **pandas_ta_args)

        # 兼容单列Series的情况
        if isinstance(pandas_result, pd.Series):
            pandas_result = pd.DataFrame({name: pandas_result})

        # 断言 pandas-ta 返回的列数量
        assert len(pandas_result.columns) == len(output_key_map), (
            f"Expected {len(output_key_map)} columns from pandas-ta, but got {len(pandas_result.columns)}."
        )

        for nb_key, pd_key in output_key_map.items():
            np_result = indicators_output_list[0][nb_key]
            pd_result = pandas_result[pd_key]

            custom_kwargs = assert_func_kwargs.get(nb_key) or assert_func_kwargs.get(
                pd_key, {}
            )

            assert_func = (
                assert_indicator_same if assert_mode else assert_indicator_different
            )
            assert_func(
                np_result,
                pd_result,
                nb_key,
                f"params {params_config}",
                **custom_kwargs,
            )


def compare_pandas_ta_with_talib(
    name,
    params_config_list,
    df_data_mock,
    input_data_keys,
    assert_mode=False,
    assert_func_kwargs={},
    output_key_maps=None,
):
    """
    通用pandas-ta和talib实现版本比较工具函数。
    name: 指标名称 (e.g., "atr", "sma")
    params_config_list: 指标参数配置列表，每个元素是一个字典，包含nb_params和pd_params
    df_data_mock: pandas DataFrame格式的OHLCV数据
    input_data_keys: 指标所需的输入数据键列表 (e.g., ["high", "low", "close"])
    assert_mode: 断言模式（True为相等，False为不相等）
    output_key_maps: 可选参数，列表格式，每个元素是一个字典，包含pandas-ta与talib输出的列名映射。
    """
    # 统一处理单列和多列的情况
    if not output_key_maps:
        # 如果是单列，创建默认的 output_key_map 列表
        output_key_maps = [{name: name} for _ in params_config_list]

    assert len(params_config_list) == len(output_key_maps), (
        "params_config_list and output_key_maps must have the same length."
    )

    for i, params_config in enumerate(params_config_list):
        pd_params_dict = params_config.get("pd_params", {})
        output_key_map = output_key_maps[i]

        _func = getattr(ta, name)

        # 动态提取输入数据
        input_series = [df_data_mock[key] for key in input_data_keys]

        # 使用字典解包传递参数给 pandas-ta
        pandas_result = _func(*input_series, **pd_params_dict, talib=False)
        talib_result = _func(*input_series, **pd_params_dict, talib=True)

        # 兼容单列Series的情况
        if isinstance(pandas_result, pd.Series):
            pandas_result = pd.DataFrame({name: pandas_result})
        if isinstance(talib_result, pd.Series):
            talib_result = pd.DataFrame({name: talib_result})

        # 断言 pandas-ta 返回的列数量
        assert len(pandas_result.columns) == len(output_key_map), (
            f"Expected {len(output_key_map)} columns from pandas-ta, but got {len(pandas_result.columns)}."
        )

        for pd_key, talib_key in output_key_map.items():
            pandas_ta_array = pandas_result[pd_key]
            talib_array = talib_result[talib_key]

            custom_kwargs = assert_func_kwargs.get(talib_key, {})
            if not custom_kwargs:
                custom_kwargs = assert_func_kwargs.get(pd_key, {})

            assert_func = (
                assert_indicator_same if assert_mode else assert_indicator_different
            )
            assert_func(
                pandas_ta_array,
                talib_array,
                pd_key,
                f"params {params_config}",
                **custom_kwargs,
            )
