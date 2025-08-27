import pandas as pd
from pathlib import Path
from src.utils.nb_convert_output import jitted_convert_all_dicts
import time


def convert_output(result, num=0, csv_path="output"):
    Path(csv_path).mkdir(parents=True, exist_ok=True)

    (
        indicators_output_list,
        signals_output_list,
        backtest_output_list,
        performance_output_list,
        indicators_output_list_mtf,
    ) = result

    (
        (indicators_keys, signals_keys, backtest_keys, indicators_keys_mtf),
        (indicators_dict, signals_dict, backtest_dict, indicators_dict_mtf),
        (
            indicators_np,
            signals_np,
            backtest_np,
            indicators_np_mtf,
        ),
        performance_keys,
        performance_dict,
        performance_value,
    ) = jitted_convert_all_dicts(
        indicators_output_list,
        signals_output_list,
        backtest_output_list,
        performance_output_list,
        indicators_output_list_mtf,
        num=num,
    )

    result_dataframe = {}
    for name, keys_item, dict_item, np_item in [
        ["indicators", indicators_keys, indicators_dict, indicators_np],
        ["signals", signals_keys, signals_dict, signals_np],
        ["backtest", backtest_keys, backtest_dict, backtest_np],
        ["indicators_mtf", indicators_keys_mtf, indicators_dict_mtf, indicators_np_mtf],
    ]:
        keys = tuple(keys_item)  # 转换List才是最快的, 如果直接转换Dict会慢
        df = pd.DataFrame(np_item, columns=keys)
        output_path = f"{csv_path}/{name}.csv"
        if csv_path:
            df.to_csv(output_path, index=False)

        result_dataframe[name] = df

    performance_result = {}
    keys = tuple(performance_keys)
    for i, k in enumerate(keys):
        performance_result[k] = float(performance_value[i])

    return result_dataframe, performance_result
