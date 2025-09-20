from numba import njit

from src.convert_output.nb_dict_item_access import get_item_from_dict_list
from src.convert_output.nb_dict_to_array_converter import convert_dict_to_np_array
from src.convert_output.nb_dict_utils import get_dict_keys_as_list

from src.utils.constants import numba_config

enable_cache = numba_config["enable_cache"]


@njit(cache=enable_cache)
def jitted_convert_all_dicts(
    params_list,
    result_list,
    num,
):
    """
    在一个 JIT 函数内，根据类型和指定索引调用特定的转换函数。
    """

    (
        tohlcv,
        indicator_params_list,
        backtest_params_list,
        tohlcv_mtf,
        indicator_params_list_mtf,
        mapping_mtf,
        tohlcv_smoothed,
        tohlcv_mtf_smoothed,
        is_only_performance,
    ) = params_list

    (
        indicators_output_list,
        signals_output_list,
        backtest_output_list,
        performance_output_list,
        indicators_output_list_mtf,
    ) = result_list

    tohlcv_dict = tohlcv
    tohlcv_mtf_dict = tohlcv_mtf
    mapping_mtf_dict = mapping_mtf
    tohlcv_smoothed_dict = tohlcv_smoothed
    tohlcv_mtf_smoothed_dict = tohlcv_mtf_smoothed

    # 从索引num中提取item
    indicator_params_dict = get_item_from_dict_list(indicator_params_list, num)
    backtest_params_dict = get_item_from_dict_list(backtest_params_list, num)
    indicator_params_mtf_dict = get_item_from_dict_list(indicator_params_list_mtf, num)

    # 把key提取成list
    indicator_params_keys = get_dict_keys_as_list(indicator_params_dict)
    backtest_params_keys = get_dict_keys_as_list(backtest_params_dict)
    indicator_params_mtf_keys = get_dict_keys_as_list(indicator_params_mtf_dict)

    # 把字典提取成1d数组
    indicator_params_np = convert_dict_to_np_array(indicator_params_dict)
    backtest_params_np = convert_dict_to_np_array(backtest_params_dict)
    indicator_params_mtf_np = convert_dict_to_np_array(indicator_params_mtf_dict)

    # 把key提取成list
    tohlcv_keys = get_dict_keys_as_list(tohlcv)
    tohlcv_mtf_keys = get_dict_keys_as_list(tohlcv_mtf)
    mapping_mtf_keys = get_dict_keys_as_list(mapping_mtf)
    tohlcv_smoothed_keys = get_dict_keys_as_list(tohlcv_smoothed)
    tohlcv_mtf_smoothed_keys = get_dict_keys_as_list(tohlcv_mtf_smoothed)

    # 把字典提取成2d数组
    tohlcv_np = convert_dict_to_np_array(tohlcv_dict)
    tohlcv_mtf_np = convert_dict_to_np_array(tohlcv_mtf_dict)
    mapping_mtf_np = convert_dict_to_np_array(mapping_mtf_dict)
    tohlcv_smoothed_np = convert_dict_to_np_array(tohlcv_smoothed_dict)
    tohlcv_mtf_smoothed_np = convert_dict_to_np_array(tohlcv_mtf_smoothed_dict)

    # 从索引num中提取item
    indicators_dict = get_item_from_dict_list(indicators_output_list, num)
    signals_dict = get_item_from_dict_list(signals_output_list, num)
    backtest_dict = get_item_from_dict_list(backtest_output_list, num)
    indicators_mtf_dict = get_item_from_dict_list(indicators_output_list_mtf, num)

    indicators_keys = get_dict_keys_as_list(indicators_dict)
    signals_keys = get_dict_keys_as_list(signals_dict)
    backtest_keys = get_dict_keys_as_list(backtest_dict)
    indicators_mtf_keys = get_dict_keys_as_list(indicators_mtf_dict)

    # 把字典提取成2d数组
    indicators_np = convert_dict_to_np_array(indicators_dict)
    signals_np = convert_dict_to_np_array(signals_dict)
    backtest_np = convert_dict_to_np_array(backtest_dict)
    indicators_mtf_np = convert_dict_to_np_array(indicators_mtf_dict)

    performance_dict = get_item_from_dict_list(performance_output_list, num)
    performance_keys = get_dict_keys_as_list(performance_dict)
    # 把字典提取成1d数组
    # performance_np = get_dict_values_as_np_array(performance_dict)
    performance_np = convert_dict_to_np_array(performance_dict)

    return (
        ("tohlcv", tohlcv_keys, tohlcv_dict, tohlcv_np),
        ("tohlcv_mtf", tohlcv_mtf_keys, tohlcv_mtf_dict, tohlcv_mtf_np),
        ("mapping_mtf", mapping_mtf_keys, mapping_mtf_dict, mapping_mtf_np),
        (
            "tohlcv_smoothed",
            tohlcv_smoothed_keys,
            tohlcv_smoothed_dict,
            tohlcv_smoothed_np,
        ),
        (
            "tohlcv_mtf_smoothed",
            tohlcv_mtf_smoothed_keys,
            tohlcv_mtf_smoothed_dict,
            tohlcv_mtf_smoothed_np,
        ),
        #
        (
            "indicator_params",
            indicator_params_keys,
            indicator_params_dict,
            indicator_params_np,
        ),
        (
            "backtest_params",
            backtest_params_keys,
            backtest_params_dict,
            backtest_params_np,
        ),
        (
            "indicator_params_mtf",
            indicator_params_mtf_keys,
            indicator_params_mtf_dict,
            indicator_params_mtf_np,
        ),
        #
        ("indicators", indicators_keys, indicators_dict, indicators_np),
        ("signals", signals_keys, signals_dict, signals_np),
        ("backtest", backtest_keys, backtest_dict, backtest_np),
        ("indicators_mtf", indicators_mtf_keys, indicators_mtf_dict, indicators_mtf_np),
        #
        ("performance", performance_keys, performance_dict, performance_np),
    )


@njit(cache=True)
def simplified_convert_results(
    result_list,
    num,
):
    """
    一个简化的 JIT 函数，用于将结果字典列表中的指定项转换为 NumPy 数组并返回。
    """

    (
        indicators_output_list,
        signals_output_list,
        backtest_output_list,
        performance_output_list,
        indicators_output_list_mtf,
    ) = result_list

    # 从索引num中提取字典项
    indicators_dict = get_item_from_dict_list(indicators_output_list, num)
    signals_dict = get_item_from_dict_list(signals_output_list, num)
    backtest_dict = get_item_from_dict_list(backtest_output_list, num)
    indicators_mtf_dict = get_item_from_dict_list(indicators_output_list_mtf, num)
    performance_dict = get_item_from_dict_list(performance_output_list, num)

    # 提取字典的键
    indicators_keys = get_dict_keys_as_list(indicators_dict)
    signals_keys = get_dict_keys_as_list(signals_dict)
    backtest_keys = get_dict_keys_as_list(backtest_dict)
    indicators_mtf_keys = get_dict_keys_as_list(indicators_mtf_dict)
    performance_keys = get_dict_keys_as_list(performance_dict)

    # 将字典转换为 NumPy 数组
    indicators_np = convert_dict_to_np_array(indicators_dict)
    signals_np = convert_dict_to_np_array(signals_dict)
    backtest_np = convert_dict_to_np_array(backtest_dict)
    indicators_mtf_np = convert_dict_to_np_array(indicators_mtf_dict)
    performance_np = convert_dict_to_np_array(performance_dict)

    return (
        ("indicators", indicators_keys, indicators_dict, indicators_np),
        ("signals", signals_keys, signals_dict, signals_np),
        ("backtest", backtest_keys, backtest_dict, backtest_np),
        ("indicators_mtf", indicators_mtf_keys, indicators_mtf_dict, indicators_mtf_np),
        ("performance", performance_keys, performance_dict, performance_np),
    )
