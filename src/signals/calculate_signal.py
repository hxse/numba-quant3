import numpy as np
import numba as nb
from numba import njit
from enum import IntEnum, auto


from .signal_0 import (
    calc_signal_0,
    get_signal_0_keys,
    get_signal_0_keys_mtf,
    # signal_0_id,
)
from .signal_1 import (
    calc_signal_1,
    get_signal_1_keys,
    get_signal_1_keys_mtf,
    # signal_1_id,
)
from .signal_2 import (
    calc_signal_2,
    get_signal_2_keys,
    get_signal_2_keys_mtf,
    # signal_2_id,
)
from .signal_3 import (
    calc_signal_3,
    get_signal_3_keys,
    get_signal_3_keys_mtf,
    # signal_3_id,
)

from src.utils.constants import numba_config
from src.utils.handle_params import convert_keys

cache = numba_config["cache"]


nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


class SignalId(IntEnum):
    signal_0_id = 0
    signal_1_id = auto()
    signal_2_id = auto()
    signal_3_id = auto()


signal_dict = {
    SignalId.signal_0_id.value: {
        "keys": convert_keys(get_signal_0_keys()),
        "keys_mtf": convert_keys(get_signal_0_keys_mtf()),
    },
    SignalId.signal_1_id.value: {
        "keys": convert_keys(get_signal_1_keys()),
        "keys_mtf": convert_keys(get_signal_1_keys_mtf()),
    },
    SignalId.signal_2_id.value: {
        "keys": convert_keys(get_signal_2_keys()),
        "keys_mtf": convert_keys(get_signal_2_keys_mtf()),
    },
    SignalId.signal_3_id.value: {
        "keys": convert_keys(get_signal_3_keys()),
        "keys_mtf": convert_keys(get_signal_3_keys_mtf()),
    },
}


@njit(cache=cache)
def calc_signal(
    signal_output,
    indicator_output,
    indicators_output_mtf,
    mapping_mtf,
    close,
    backtest_params,
):
    signal_select = backtest_params["signal_select"]

    if signal_select == SignalId.signal_0_id.value:
        calc_signal_0(
            signal_output,
            indicator_output,
            indicators_output_mtf,
            mapping_mtf,
            close,
        )
    elif signal_select == SignalId.signal_1_id.value:
        calc_signal_1(
            signal_output,
            indicator_output,
            indicators_output_mtf,
            mapping_mtf,
            close,
        )
    elif signal_select == SignalId.signal_2_id.value:
        calc_signal_2(
            signal_output,
            indicator_output,
            indicators_output_mtf,
            mapping_mtf,
            close,
        )
    elif signal_select == SignalId.signal_3_id.value:
        calc_signal_3(
            signal_output,
            indicator_output,
            indicators_output_mtf,
            mapping_mtf,
            close,
        )
