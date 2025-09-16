import pandas as pd
from pathlib import Path
from src.utils.nb_convert_output import jitted_convert_all_dicts
import time


def convert_output(params, result, num=0, output_path="output"):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    params_tuple = tuple(params)
    result_tuple = tuple(result)

    result_converted = jitted_convert_all_dicts(params_tuple, result_tuple, num)

    final_result = {}
    for name, keys_item, dict_item, np_item in result_converted:
        # 用List转换成tuple才是最快的, 如果用Dict转换成tuple会慢
        keys = tuple(keys_item)

        if len(np_item.shape) == 1:
            final_result[name] = {}
            for i, k in enumerate(keys):
                final_result[name][k] = float(np_item[i])
        elif len(np_item.shape) == 2:
            df = pd.DataFrame(np_item, columns=keys)
            final_result[name] = df
        else:
            raise RuntimeError(f"检测到未预期维度数 {len(np_item).shape}")

        # _output_path = f"{output_path}/{name}.csv"
        # _output_path = f"{output_path}/{name}.parquet"
        # if _output_path:
        #     # df.to_csv(_output_path, index=False)
        #     df.to_parquet(_output_path, index=False)

    return final_result
