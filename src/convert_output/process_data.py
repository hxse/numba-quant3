import polars as pl
from src.convert_output.converter import convert_nb_data_to_py_dicts
from typing import Any
import numpy as np


def to_serializable_and_polars_df(data: Any) -> Any:
    """
    递归地转换数据结构，将 NumPy 标量转换为 Python 原生类型，
    并将符合条件的字典转换为 Polars DataFrame。

    参数:
    ----------
    data : Any
        要转换的数据，可以是字典、列表或任意其他类型。

    返回:
    -------
    Any
        转换后的数据。
    """
    # 1. 如果数据是字典，首先检查是否能转换为 Polars DataFrame
    if isinstance(data, dict):
        is_convertible_to_df = True
        # 检查是否所有键都是字符串，所有值都是 NumPy 数组
        for key, value in data.items():
            if not isinstance(key, str) or not isinstance(value, np.ndarray):
                is_convertible_to_df = False
                break

        # 如果所有条件都满足，则转换为 DataFrame
        if is_convertible_to_df:
            return pl.DataFrame(data)

        # 如果不能转换为 DataFrame，递归处理字典中的值
        return {
            key: to_serializable_and_polars_df(value) for key, value in data.items()
        }

    # 2. 如果数据是列表，递归处理其元素
    if isinstance(data, list):
        return [to_serializable_and_polars_df(element) for element in data]

    # 3. 如果数据是 NumPy 标量，将其转换为 Python 原生类型
    if isinstance(data, np.generic):
        return data.item()

    # 4. 其他类型，直接返回
    return data


def process_data_output(
    params: tuple,
    data_list: tuple,
    convert_num: int = 0,
    data_suffix: str = ".csv",
    params_suffix: str = ".json",
):
    """
    转换数据并处理输出。
    upload_server用127.0.0.1, 别用localhost, 会慢
    """

    # 把数据从nb_data转换成py_dict
    result_converted = convert_nb_data_to_py_dicts(params, data_list, convert_num)

    # 把第num个提取出来
    for k, v in result_converted.items():
        if k in ["ohlcv_mtf", "ohlcv_smoothed_mtf", "data_mapping"]:
            continue

        if isinstance(k, str) and isinstance(v, list):
            assert 0 <= convert_num < len(v), f"检测到num越界 {convert_num} {len(v)}"
            result_converted[k] = v[convert_num]

    # 把numpy标量转成py标量
    result_converted = to_serializable_and_polars_df(result_converted)

    def create_named_object(number, k, item):
        if isinstance(item, dict):
            if number is None:
                return {"name": f"{k}{params_suffix}", "data": item}
            return {"name": f"{k}_{number}{params_suffix}", "data": item}
        elif isinstance(item, pl.DataFrame):
            if number is None:
                return {"name": f"{k}{data_suffix}", "data": item}
            return {"name": f"{k}_{number}{data_suffix}", "data": item}

    data_list = []
    for k, v in result_converted.items():
        if isinstance(v, list):
            for number, item in enumerate(v):
                data = create_named_object(number, k, item)
                if data:
                    data_list.append(data)
        else:
            data = create_named_object(None, k, v)
            if data:
                data_list.append(data)

    return result_converted, data_list
