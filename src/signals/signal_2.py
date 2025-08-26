from numba import njit, types
from numba.typed import List


from src.utils.constants import numba_config


from src.utils.nb_check_keys import check_all

cache = numba_config["cache"]

signal_2_id = 2


@njit(cache=cache)
def get_signal_2_keys():
    _l = List.empty_list(types.unicode_type)
    for i in ("sma", "sma2"):
        _l.append(i)
    return _l


@njit(cache=cache)
def get_signal_2_keys_mtf():
    _l = List.empty_list(types.unicode_type)
    for i in ("",):
        _l.append(i)
    return _l


@njit(cache=cache)
def calc_signal_2(
    _tohlcv,
    _tohlcv_mtf,
    data_mapping,
    indicator_output,
    indicators_output_mtf,
    signal_output,
):
    if not check_all(
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
