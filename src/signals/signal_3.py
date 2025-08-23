from numba import njit, types
from numba.typed import List


from src.utils.constants import numba_config


from src.utils.nb_check_keys import check_all

cache = numba_config["cache"]

signal_3_id = 3


@njit(cache=cache)
def get_signal_3_keys():
    _l = List.empty_list(types.unicode_type)
    for i in ("bbands_upper", "bbands_middle", "bbands_lower"):
        _l.append(i)
    return _l


@njit(cache=cache)
def get_signal_3_keys_mtf():
    _l = List.empty_list(types.unicode_type)
    for i in ("sma", "sma2"):
        _l.append(i)
    return _l


@njit(cache=cache)
def calc_signal_3(
    _tohlcv,
    _tohlcv_mtf,
    mapping_mtf,
    indicator_output,
    indicators_output_mtf,
    signal_output,
):
    exist_key = check_all(
        _tohlcv,
        _tohlcv_mtf,
        get_signal_3_keys(),
        get_signal_3_keys_mtf(),
        indicator_output,
        indicators_output_mtf,
        mapping_mtf,
    )
    if not exist_key:
        return

    if exist_key:
        close = _tohlcv["close"]
        sma = indicators_output_mtf["sma"][mapping_mtf["mtf"]]
        sma2 = indicators_output_mtf["sma2"][mapping_mtf["mtf"]]
        bbands_upper = indicator_output["bbands_upper"]
        bbands_middle = indicator_output["bbands_middle"]
        bbands_lower = indicator_output["bbands_lower"]

        signal_output["enter_long"] = (close < bbands_lower) & (sma > bbands_middle)
        signal_output["exit_long"] = close > bbands_middle
        signal_output["enter_short"] = (close > bbands_upper) & (sma < bbands_middle)
        signal_output["exit_short"] = close < bbands_middle
