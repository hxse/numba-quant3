from numba import njit, types
from numba.typed import List


from src.utils.constants import numba_config


from src.utils.nb_check_keys import check_all


cache = numba_config["cache"]


signal_0_id = 0


@njit(cache=cache)
def get_signal_0_keys():
    _l = List.empty_list(types.unicode_type)
    return _l


@njit(cache=cache)
def get_signal_0_keys_mtf():
    _l = List.empty_list(types.unicode_type)  # 如果只用大周期tohlcv数据,可以用""占位
    return _l


@njit(cache=cache)
def calc_signal_0(
    _tohlcv,
    _tohlcv_mtf,
    mapping_mtf,
    indicator_output,
    indicators_output_mtf,
    signal_output,
):
    """
    calc_signal_0是占位的空函数
    """

    exist_key = check_all(
        _tohlcv,
        _tohlcv_mtf,
        get_signal_0_keys(),
        get_signal_0_keys_mtf(),
        indicator_output,
        indicators_output_mtf,
        mapping_mtf,
    )
    if not exist_key:
        return
