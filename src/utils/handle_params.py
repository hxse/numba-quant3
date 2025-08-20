import numpy as np
import numba as nb

# 从全局配置中获取 Numba 类型
from src.utils.constants import numba_config

cache = numba_config["cache"]
nb_float = numba_config["nb"]["float"]
np_float = numba_config["np"]["float"]

print("params cache", cache)


from .nb_params import create_params_list_template, set_params_list_value


def convert_keys(keys):
    # 使用列表推导式获取所有元素
    split_keys = [i.split("_")[0] for i in keys]

    # 使用 dict.fromkeys() 高效去重并保留顺序
    unique_list = list(dict.fromkeys(split_keys))

    # 将结果转换为 tuple 并返回
    return tuple(unique_list)


def init_params_with_enable(
    params_count,
    signal_select_id,
    signal_dict,
    enable_large,
):
    keys = "keys_large" if enable_large else "keys"

    (indicator_params_list, backtest_params_list) = create_params_list_template(
        params_count
    )

    set_params_list_value(
        "signal_select",
        backtest_params_list,
        np.array([signal_select_id for i in range(params_count)], dtype=np_float),
    )

    signal_keys = signal_dict[signal_select_id][keys]

    for i in signal_keys:
        key = f"{i}_enable"
        target_array = np.array([True for i in range(params_count)], dtype=np_float)

        set_params_list_value(
            key,
            indicator_params_list,
            target_array,
        )
    return indicator_params_list, backtest_params_list
