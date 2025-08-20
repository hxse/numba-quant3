from numba import njit, types
from numba.typed import List


from src.utils.constants import numba_config


from src.utils.nb_check_keys import check_keys, check_mapping, check_all

cache = numba_config["cache"]

signal_2_id = 2
signal_2_keys = ("bbands_upper", "bbands_middle", "bbands_lower")
signal_2_keys_mtf = ("sma", "sma2")
# signal_2_keys_large = ()


@njit(cache=cache)
def calc_signal_2(
    signal_output, indicator_output, indicators_output_mtf, mapping_mtf, close
):
    exist_key = check_all(
        len(close),
        signal_2_keys,
        signal_2_keys_mtf,
        indicator_output,
        indicators_output_mtf,
        mapping_mtf,
    )
    if not exist_key:
        return

    if exist_key:
        bbands_upper = indicator_output["bbands_upper"]
        bbands_middle = indicator_output["bbands_middle"]
        bbands_lower = indicator_output["bbands_lower"]
        signal_output["enter_long"] = close < bbands_lower
        signal_output["exit_long"] = close > bbands_middle
        signal_output["enter_short"] = close > bbands_upper
        signal_output["exit_short"] = close < bbands_middle
