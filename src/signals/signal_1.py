from numba import njit, types
from numba.typed import List


from src.utils.constants import numba_config


from src.utils.nb_check_keys import check_keys, check_mapping, check_all


cache = numba_config["cache"]

signal_1_id = 1
signal_1_keys = ("sma", "sma2")
signal_1_keys_mtf = ()


@njit(cache=cache)
def calc_signal_1(
    signal_output, indicator_output, indicators_output_mtf, mapping_mtf, close
):
    exist_key = check_all(
        len(close),
        signal_1_keys,
        signal_1_keys_mtf,
        indicator_output,
        indicators_output_mtf,
        mapping_mtf,
    )
    if not exist_key:
        return

    sma = indicator_output["sma"]
    sma2 = indicator_output["sma2"]
    signal_output["enter_long"] = sma > sma2
    signal_output["exit_long"] = sma < sma2
    signal_output["enter_short"] = sma < sma2
    signal_output["exit_short"] = sma > sma2
