from numba import njit, types
from numba.typed import List


from src.utils.constants import numba_config


from src.utils.nb_check_keys import check_keys, check_mapping, check_all


cache = numba_config["cache"]


signal_0_id = 0
signal_0_keys = ()
signal_0_keys_mtf = ()


@njit(cache=cache)
def calc_signal_0(
    signal_output, indicator_output, indicators_output_mtf, mapping_mtf, close
):
    """
    calc_signal_0是占位的空函数
    """

    exist_key = check_all(
        len(close),
        signal_0_keys,
        signal_0_keys_mtf,
        indicator_output,
        indicators_output_mtf,
        mapping_mtf,
    )
    if not exist_key:
        return
