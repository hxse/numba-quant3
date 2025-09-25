from numba import njit, types
from numba.typed import List
from numba.core.types import unicode_type
from src.utils.constants import numba_config


from src.utils.nb_check_keys import check_data_for_signal

enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]

signal_3_id = 3


# 定义字典类型：使用 types.DictType
param_dict_type = types.DictType(unicode_type, nb_float)

# 定义列表类型：使用 types.ListType
params_list_type = types.ListType(unicode_type)


@njit(cache=enable_cache)
def get_i_output_mtf_need_keys():
    outer_list = List.empty_list(List.empty_list(types.unicode_type))

    # 创建并填充内部列表
    inner_list_1 = List.empty_list(types.unicode_type)
    for i in ("bbands_upper", "bbands_middle", "bbands_lower"):
        inner_list_1.append(i)
    outer_list.append(inner_list_1)

    # 创建并填充内部列表
    inner_list_2 = List.empty_list(types.unicode_type)
    for i in ("sma", "sma2"):
        inner_list_2.append(i)
    outer_list.append(inner_list_2)

    # 将内部列表添加到外部列表中
    return outer_list


@njit(cache=enable_cache)
def get_signal_3_keys():
    _l = List.empty_list(types.unicode_type)
    for i in ("bbands_upper", "bbands_middle", "bbands_lower"):
        _l.append(i)
    return _l


@njit(cache=enable_cache)
def get_signal_3_keys_mtf():
    _l = List.empty_list(types.unicode_type)
    for i in ("sma", "sma2"):
        _l.append(i)
    return _l


@njit(cache=enable_cache)
def calc_signal_3(
    ohlcv_mtf,
    data_mapping,
    i_output_mtf,
    s_output,
):
    if not check_data_for_signal(
        ohlcv_mtf, get_i_output_mtf_need_keys(), i_output_mtf, data_mapping
    ):
        return

    ohlcv_a = ohlcv_mtf[0]
    close = ohlcv_a["close"]

    i_output_a = i_output_mtf[0]
    i_output_b = i_output_mtf[1]

    bbands_upper = i_output_a["bbands_upper"]
    bbands_middle = i_output_a["bbands_middle"]
    bbands_lower = i_output_a["bbands_lower"]

    mtf_b = data_mapping["mtf_1"]

    sma = i_output_b["sma"][mtf_b]
    sma2 = i_output_b["sma2"][mtf_b]

    s_output["enter_long"] = (close < bbands_lower) & (sma > bbands_middle)
    s_output["exit_long"] = close > bbands_middle
    s_output["enter_short"] = (close > bbands_upper) & (sma < bbands_middle)
    s_output["exit_short"] = close < bbands_middle

    skip = data_mapping["skip"]
    s_output["enter_long"][skip == 0] = False
    s_output["enter_short"][skip == 0] = False
