import numpy as np
import numba as nb

# 从全局配置中获取 Numba 类型
from src.utils.constants import numba_config

cache = numba_config["cache"]
nb_float = numba_config["nb"]["float"]
np_float = numba_config["np"]["float"]


from .nb_params import (
    create_params_list_template,
    set_params_list_value,
    get_data_mapping,
    init_tohlcv,
    init_tohlcv_smoothed,
)


def convert_keys(keys):
    # 使用列表推导式获取所有元素
    split_keys = [i.split("_")[0] for i in keys]

    # 使用 dict.fromkeys() 高效去重并保留顺序
    unique_list = list(dict.fromkeys(split_keys))

    # 将结果转换为 tuple 并返回
    return tuple(unique_list)


# 年化因子, 加密货币
annualization_factor_dict = {
    # 基于总分钟数计算
    "1m": 365 * 24 * 60 / 1,
    "3m": 365 * 24 * 60 / 3,
    "5m": 365 * 24 * 60 / 5,
    "10m": 365 * 24 * 60 / 10,
    "15m": 365 * 24 * 60 / 15,
    "30m": 365 * 24 * 60 / 30,
    # 基于总小时数计算
    "1h": 365 * 24 / 1,
    "4h": 365 * 24 / 4,
    "6h": 365 * 24 / 6,
    "12h": 365 * 24 / 12,
    # 基于总天数计算
    "1d": 365 / 1,
    # 每周
    "1w": 365 / 7,
    # 每月
    "1M": 12,
}


def init_params(
    params_count,
    signal_select_id,
    signal_dict,
    tohlcv_np,
    tohlcv_np_mtf=None,
    mapping_mtf=None,
    smooth_mode=None,
    period=None,
    is_only_performance=False,
):
    """
    三个mtf参数: tohlcv_np_mtf, indicator_params_list_mtf, mapping_mtf
    如果keys_mtf是(), 那么三个mtf参数都会被设为None
    如果keys_mtf是(""),三个mtf参数都正常,只不过indicator_params_list_mtf不会有任何enable,需要tohlcv_np_mtf数据
    如果keys_mtf是("sma")三个mtf参数都正常,indicator_params_list_mtf中的sma_enable会被打开, 需要tohlcv_np_mtf数据
    """

    result = []
    for keys in ["keys", "keys_mtf"]:
        signal_keys = signal_dict[signal_select_id][keys]

        (indicator_params_list, backtest_params_list) = create_params_list_template(
            params_count, empty=False
        )

        set_params_list_value(
            "signal_select",
            backtest_params_list,
            np.array([signal_select_id for i in range(params_count)], dtype=np_float),
        )

        for i in signal_keys:
            if i == "":
                continue
            key = f"{i}_enable"
            target_array = np.array([True for i in range(params_count)], dtype=np_float)

            set_params_list_value(
                key,
                indicator_params_list,
                target_array,
            )
        result.append(
            {
                "indicator_params_list": indicator_params_list,
                "backtest_params_list": backtest_params_list,
            }
        )

    # 更新年化因子
    if period:
        set_params_list_value(
            "annualization_factor",
            result[0]["backtest_params_list"],
            np.array(
                [annualization_factor_dict[period] for i in range(params_count)],
                dtype=np_float,
            ),
        )

    assert tohlcv_np is not None, "小周期数据不能为none"

    indicator_params_list_mtf = result[1]["indicator_params_list"]
    signal_keys = signal_dict[signal_select_id]["keys_mtf"]
    if len(signal_keys) == 0:
        tohlcv_mtf = init_tohlcv(None)
        indicator_params_list_mtf = create_params_list_template(
            params_count, empty=True
        )[0]
        mapping_mtf = get_data_mapping(None, None)
    else:
        assert tohlcv_np_mtf is not None, "大周期数据不能为none"
        tohlcv_mtf = init_tohlcv(tohlcv_np_mtf)
        mapping_mtf = get_data_mapping(tohlcv_np, tohlcv_np_mtf)

    tohlcv = init_tohlcv(tohlcv_np)
    tohlcv_smoothed = init_tohlcv_smoothed(tohlcv_np, smooth_mode=smooth_mode)
    tohlcv_mtf_smoothed = init_tohlcv_smoothed(tohlcv_np_mtf, smooth_mode=smooth_mode)

    result_dict = {
        "tohlcv": tohlcv,
        "indicator_params_list": result[0]["indicator_params_list"],
        "backtest_params_list": result[0]["backtest_params_list"],
        "tohlcv_mtf": tohlcv_mtf,
        "indicator_params_list_mtf": indicator_params_list_mtf,
        "mapping_mtf": mapping_mtf,
        "tohlcv_smoothed": tohlcv_smoothed,
        "tohlcv_mtf_smoothed": tohlcv_mtf_smoothed,
        "is_only_performance": is_only_performance,
    }

    return tuple(result_dict.values())
