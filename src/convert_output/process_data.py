import polars as pl
from src.convert_output.nb_main_converter import jitted_convert_all_dicts


def process_data_output(
    params: tuple,
    result: tuple,
    num: int = 0,
    data_suffix: str = ".csv",
    params_suffix: str = ".json",
):
    """
    转换数据并处理输出。
    upload_server用127.0.0.1, 别用localhost, 会慢
    """

    result_converted = jitted_convert_all_dicts(params, result, num)
    final_result = {}
    data_list = []

    for name, keys_item, dict_item, np_item in result_converted:
        keys = tuple(keys_item)
        if len(np_item.shape) == 1:
            _dict = {k: float(v) for k, v in zip(keys, np_item)}
            final_result[name] = _dict
            data_list.append((f"{name}{params_suffix}", _dict))
        elif len(np_item.shape) == 2:
            df = pl.from_numpy(np_item, schema=keys)
            final_result[name] = df
            data_list.append((f"{name}{data_suffix}", df))
        else:
            raise RuntimeError(f"检测到未预期维度数 {len(np_item.shape)}")

    return final_result, data_list
