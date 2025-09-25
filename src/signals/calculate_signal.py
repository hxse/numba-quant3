import numpy as np
import numba as nb
from numba import njit, types
from numba.typed import Dict, List
from enum import IntEnum, auto

from parallel_signature import signal_signature


from .signal_0 import (
    calc_signal_0,
    get_signal_0_keys,
    get_signal_0_keys_mtf,
    get_signal_0_keys_test,
    # signal_0_id,
)
from .signal_1 import (
    calc_signal_1,
    get_signal_1_keys,
    get_signal_1_keys_mtf,
    get_signal_1_keys_test,
    # signal_1_id,
)
from .signal_2 import (
    calc_signal_2,
    get_signal_2_keys,
    get_signal_2_keys_mtf,
    get_signal_2_keys_test,
    # signal_2_id,
)
from .signal_3 import (
    calc_signal_3,
    get_signal_3_keys,
    get_signal_3_keys_mtf,
    get_i_output_mtf_need_keys,
    # signal_3_id,
)

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


# import time

# """
# 参考 minimal_example\numba_jit_warmup_test.py
# numba最好用非空数据来预热, 而且1d,2d不同形状都要预热
# """
# signal_keys = get_signal_3_keys()
# signal_keys_mtf = get_signal_3_keys_mtf()

# _ = convert_keys(get_list_one(), False)
# _ = convert_keys(get_2d_list_one(), False)


# start_time = time.perf_counter()

# print(convert_keys(signal_keys), False)
# print(convert_keys(signal_keys), True)
# print(convert_keys(signal_keys_mtf), False)
# print(convert_keys(signal_keys_mtf), True)

# print(f"test 转换keys 耗时{time.perf_counter() - start_time}")


# import pdb

# pdb.set_trace()


class SignalId(IntEnum):
    signal_0_id = 0
    signal_1_id = auto()
    signal_2_id = auto()
    signal_3_id = auto()


signal_dict = {
    SignalId.signal_0_id.value: {
        "keys": convert_keys(get_signal_0_keys_test(), is_split=True),
    },
    SignalId.signal_1_id.value: {
        "keys": convert_keys(get_signal_1_keys_test(), is_split=True),
    },
    SignalId.signal_2_id.value: {
        "keys": convert_keys(get_signal_2_keys_test(), is_split=True),
    },
    SignalId.signal_3_id.value: {
        "keys": convert_keys(get_i_output_mtf_need_keys(), is_split=True),
    },
}


@njit(signal_signature, cache=enable_cache)
def calc_signal(ohlcv_mtf, data_mapping, i_output_mtf, s_output, b_params):
    signal_select = b_params["signal_select"]

    # if signal_select == SignalId.signal_0_id.value:
    #     calc_signal_0(
    #         ohlcv_mtf,
    #         data_mapping,
    #         i_output_mtf,
    #         s_output,
    #     )
    # elif signal_select == SignalId.signal_1_id.value:
    #     calc_signal_1(
    #         ohlcv_mtf,
    #         data_mapping,
    #         i_output_mtf,
    #         s_output,
    #     )
    # elif signal_select == SignalId.signal_2_id.value:
    #     calc_signal_2(
    #         ohlcv_mtf,
    #         data_mapping,
    #         i_output_mtf,
    #         s_output,
    #     )

    if signal_select == SignalId.signal_3_id.value:
        calc_signal_3(
            ohlcv_mtf,
            data_mapping,
            i_output_mtf,
            s_output,
        )
