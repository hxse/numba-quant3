import numpy as np
from typing import List, Dict
import numpy.typing as npt

from src.convert_params.annualization_calculator import get_annualization_factor
from src.convert_params.param_template_manager import (
    set_params_list_value,
    set_params_list_value_mtf,
    create_indicator_params_list,
    create_backtest_params_list,
)
from src.convert_params.data_preprocessor import (
    init_tohlcv,
    init_tohlcv_smoothed,
    get_data_mapping_mtf,
)
from src.convert_params.param_key_utils import (
    get_item_from_list,
    get_item_from_2d_list,
    create_list_dict_float_1d_empty,
    create_list_dict_float_1d_one,
    create_dict_float_1d_empty,
    create_dict_float_1d_one,
    append_item,
    get_length_from_list_or_dict,
)
import time

from src.utils.constants import numba_config


nb_float = numba_config["nb"]["float"]
np_float = numba_config["np"]["float"]


def init_params(
    params_count: int,
    signal_select_id: int,
    signal_dict: Dict[int, Dict[str, List[List[str]]]],
    ohlcv_mtf_np_list: List[npt.NDArray[np.generic]],
    period_list: list[str],
    smooth_mode: str = "",
    is_only_performance: bool = False,
):
    """
    三个mtf参数: ohlcv_mtf_np, indicator_params_list_mtf, mapping_mtf
    如果keys_mtf是(), 那么三个mtf参数都会被设为None
    如果keys_mtf是(""),三个mtf参数都正常,只不过indicator_params_list_mtf不会有任何enable,需要ohlcv_mtf_np数据
    如果keys_mtf是("sma")三个mtf参数都正常,indicator_params_list_mtf中的sma_enable会被打开, 需要ohlcv_mtf_np数据
    """
    # ---- 处理数据 ----

    smooth_mode = "" if not smooth_mode else smooth_mode

    ohlcv_mtf_np_list = [i for i in ohlcv_mtf_np_list if i is not None]

    signal_keys = signal_dict[signal_select_id]["keys"]

    assert len(ohlcv_mtf_np_list) == len(signal_keys), (
        f"mtf多时间周期数据不匹配 {len(ohlcv_mtf_np_list)} {len(signal_keys)}"
    )

    ohlcv_mtf = create_list_dict_float_1d_empty()
    for i in ohlcv_mtf_np_list:
        append_item(ohlcv_mtf, init_tohlcv(i))

    data_mapping = get_data_mapping_mtf(ohlcv_mtf)

    ohlcv_smoothed_mtf = create_list_dict_float_1d_empty()
    for i in ohlcv_mtf_np_list:
        ohlcv_smoothed = init_tohlcv_smoothed(i, smooth_mode=smooth_mode)
        if get_length_from_list_or_dict(ohlcv_smoothed) > 0:
            append_item(ohlcv_smoothed_mtf, ohlcv_smoothed)

    assert (get_length_from_list_or_dict(ohlcv_smoothed_mtf) == 0) or (
        get_length_from_list_or_dict(ohlcv_smoothed_mtf)
        == get_length_from_list_or_dict(ohlcv_mtf)
    ), "需要ohlcv_smoothed_mtf长度等于0, 或者等于ohlcv_mtf长度"

    # ---- 处理参数 ----

    backtest_params = create_backtest_params_list(params_count, empty=False)

    set_params_list_value(
        "signal_select",
        backtest_params,
        np.full((params_count,), signal_select_id, dtype=np_float),
    )

    indicator_params_mtf = create_indicator_params_list(
        params_count, len(signal_keys), empty=False
    )

    for idx, keys_array in enumerate(signal_keys):
        for name in keys_array:
            key = f"{name}_enable"
            target_array = np.full((params_count,), True, dtype=np_float)
            set_params_list_value_mtf(idx, key, indicator_params_mtf, target_array)

    # ---- 年化因子 ----

    assert isinstance(period_list, list), (
        f"period_list must be list, but got {type(period_list)}"
    )
    assert all(isinstance(item, str) for item in period_list), (
        "All items in period_list must be strings."
    )
    assert len(period_list) > 0, f"period length must > 1, but got {len(period_list)}"
    annualization_factor = get_annualization_factor(period_list[0])
    set_params_list_value(
        "annualization_factor",
        backtest_params,
        np.full((params_count,), annualization_factor, dtype=np_float),
    )

    result_dict = {
        "ohlcv_mtf": ohlcv_mtf,
        "ohlcv_smoothed_mtf": ohlcv_smoothed_mtf,
        "data_mapping": data_mapping,
        "indicator_params_mtf": indicator_params_mtf,
        "backtest_params": backtest_params,
        "is_only_performance": is_only_performance,
    }

    return tuple(result_dict.values())
