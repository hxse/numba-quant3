from numba import njit, types
from numba.typed import List


from src.utils.constants import numba_config


from src.utils.nb_check_keys import check_data_for_signal

enable_cache = numba_config["enable_cache"]

signal_2_id = 2


@njit(cache=enable_cache)
def get_signal_2_keys_test():
    outer_list = List.empty_list(List.empty_list(types.unicode_type))

    # 创建并填充内部列表
    inner_list_1 = List.empty_list(types.unicode_type)
    for i in ("sma", "sma2"):
        inner_list_1.append(i)
    outer_list.append(inner_list_1)

    # 创建并填充内部列表
    inner_list_2 = List.empty_list(types.unicode_type)
    for i in ("",):
        inner_list_2.append(i)
    outer_list.append(inner_list_2)

    # 将内部列表添加到外部列表中
    return outer_list


@njit(cache=enable_cache)
def get_signal_2_keys():
    _l = List.empty_list(types.unicode_type)
    for i in ("sma", "sma2"):
        _l.append(i)
    return _l


@njit(cache=enable_cache)
def get_signal_2_keys_mtf():
    _l = List.empty_list(types.unicode_type)
    for i in ("",):
        _l.append(i)
    return _l


@njit(cache=enable_cache)
def calc_signal_2(
    _tohlcv,
    _tohlcv_mtf,
    data_mapping,
    indicator_output,
    indicators_output_mtf,
    signal_output,
):
    if not check_data_for_signal(
        _tohlcv,
        _tohlcv_mtf,
        get_signal_2_keys(),
        get_signal_2_keys_mtf(),
        indicator_output,
        indicators_output_mtf,
        data_mapping,
    ):
        return

    sma = indicator_output["sma"]
    sma2 = indicator_output["sma2"]
    signal_output["enter_long"] = sma > sma2
    signal_output["exit_long"] = sma < sma2
    signal_output["enter_short"] = sma < sma2
    signal_output["exit_short"] = sma > sma2

    skip = data_mapping["skip"]
    signal_output["enter_long"][skip == 0] = False
    signal_output["enter_short"][skip == 0] = False
