import numpy as np
import numba as nb
from numba import njit, types
from numba.core.types import unicode_type
from numba.typed import Dict, List
from enum import IntEnum, auto


from parallel_signature import signal_signature


from .signal_0 import calc_signal_0, define_signal_0_params
from .signal_1 import calc_signal_1, define_signal_1_params
from .signal_2 import calc_signal_2, define_signal_2_params
from .signal_3 import calc_signal_3, define_signal_3_params

from src.utils.constants import numba_config
from src.convert_params.param_key_utils import (
    convert_keys,
    get_length_from_list_or_dict,
    create_list_unicode_empty,
    create_2d_list_unicode_empty,
    create_list_unicode_one,
    create_2d_list_unicode_one,
    get_item_from_list,
)

enable_cache = numba_config["enable_cache"]


nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


class SignalId(IntEnum):
    signal_0_id = 0
    signal_1_id = auto()
    signal_2_id = auto()
    signal_3_id = auto()


si = SignalId

signal_dict = {
    si.signal_0_id.value: {
        "keys": convert_keys(define_signal_0_params()),
    },
    si.signal_1_id.value: {
        "keys": convert_keys(define_signal_1_params()),
    },
    si.signal_2_id.value: {
        "keys": convert_keys(define_signal_2_params()),
    },
    si.signal_3_id.value: {
        "keys": convert_keys(define_signal_3_params()),
    },
}


@njit(signal_signature, cache=enable_cache)
def calc_signal(ohlcv_mtf, data_mapping, i_output_mtf, s_output, b_params):
    signal_select = b_params["signal_select"]

    if signal_select == si.signal_0_id.value:
        calc_signal_0(
            ohlcv_mtf,
            data_mapping,
            i_output_mtf,
            s_output,
        )
    if signal_select == si.signal_1_id.value:
        calc_signal_1(
            ohlcv_mtf,
            data_mapping,
            i_output_mtf,
            s_output,
        )
    elif signal_select == si.signal_2_id.value:
        calc_signal_2(
            ohlcv_mtf,
            data_mapping,
            i_output_mtf,
            s_output,
        )
    elif signal_select == si.signal_3_id.value:
        calc_signal_3(
            ohlcv_mtf,
            data_mapping,
            i_output_mtf,
            s_output,
        )
