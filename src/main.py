import sys
from pathlib import Path

root_path = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()),
    None,
)
if root_path:
    sys.path.insert(0, str(root_path))

import numpy as np
import time
from src.utils.typer_tool import typer, app
from src.utils.constants import numba_config, set_numba_dtypes


def main(
    cache: bool = typer.Option(True, "--cache/--no-cache", help="启用或禁用Numba缓存"),
    enable64: bool = typer.Option(True, "--64bit/--32bit", help="启用或禁用64位浮点数"),
    show_timing: bool = typer.Option(True, "--timing", help="启用或禁用运行时间打印。"),
):
    """
    执行整个回测流程并可选地打印运行时间。
    """
    # 记录整个main函数开始时间
    if show_timing:
        main_start_time = time.time()

    # 更新全局配置字典
    set_numba_dtypes(numba_config, enable64=enable64, cache=cache)
    print(f"cache from cli: {cache}")

    cache = numba_config["cache"]
    nb_float = numba_config["nb"]["float"]
    np_float = numba_config["np"]["float"]

    # 在解析参数并更新全局配置字典后，再导入 Numba 函数
    from params import (
        create_params_list_template,
        create_params_dict_template,
        get_params_list_value,
        set_params_list_value,
        get_params_dict_value,
        set_params_dict_value,
        convert_params_dict_list,
    )
    from utils.mock_data import get_mock_data
    from parallel import parallel_entry

    # 记录参数生成开始时间，并计算冷启动时间
    if show_timing:
        params_start_time = time.time()
        cold_start_duration = params_start_time - main_start_time
        print(f"\n冷启动时间 (进入main到参数生成前): {cold_start_duration:.4f} 秒")

    tohlcv_np = get_mock_data(data_count=3)

    (indicator_params_list, backtest_params_list) = create_params_list_template(
        params_count=2
    )
    np_arr = get_params_list_value("sma_period", indicator_params_list)
    indicator_params_list = set_params_list_value(
        "sma_period", indicator_params_list, np.array([1, 2], dtype=np_float)
    )

    (indicator_params_dict, backtest_params_dict) = create_params_dict_template(
        params_count=2
    )
    _indicator_params_list = convert_params_dict_list(indicator_params_dict)

    _indicator_params_dict = get_params_dict_value("sma_period", indicator_params_dict)
    _indicator_params_dict2 = set_params_dict_value(
        "sma_period", indicator_params_dict, np.array([1, 2], dtype=np_float)
    )
    import pdb

    pdb.set_trace()

    # 记录参数生成结束时间，并计算运行时间
    if show_timing:
        params_end_time = time.time()
        params_duration = params_end_time - params_start_time
        print(f"参数生成时间 (default_params): {params_duration:.4f} 秒")

    # 记录 parallel_entry 函数开始时间
    if show_timing:
        parallel_start_time = time.time()

    result = parallel_entry(tohlcv_np, indicator_params_list, backtest_params_list)
    (indicators_list, signals_list, backtest_list, performance_list) = result

    # 记录 parallel_entry 函数结束时间并打印内核运行时间
    if show_timing:
        parallel_end_time = time.time()
        parallel_duration = parallel_end_time - parallel_start_time
        print(f"parallel_entry 内核运行时间: {parallel_duration:.4f} 秒")

    print(f"\nNumba 函数结果: {len(indicators_list)}")

    # 记录整个main函数结束时间并打印总运行时间
    if show_timing:
        main_end_time = time.time()
        main_duration = main_end_time - main_start_time
        print(f"main 函数总运行时间: {main_duration:.4f} 秒")


if __name__ == "__main__":
    app.command("main | ma")(main)
    app()
