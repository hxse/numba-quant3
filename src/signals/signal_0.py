from numba import njit, types
from numba.typed import List


from src.utils.constants import numba_config


from src.utils.nb_check_keys import check_keys


cache = numba_config["cache"]


signal_0_id = 0
signal_0_keys = ()


@njit(cache=cache)
def calc_signal_0(signal_item, indicator_item, close):
    """
    calc_signal_0是占位的空函数
    """
    exist_key = check_keys(signal_0_keys, indicator_item)
    if not exist_key:
        return
