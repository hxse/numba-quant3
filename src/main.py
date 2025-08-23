import time

base_start_time = time.time()

import sys
from pathlib import Path

root_path = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()),
    None,
)
if root_path:
    sys.path.insert(0, str(root_path))

import numpy as np
from src.utils.typer_tool import typer, app
from src.utils.constants import numba_config, set_numba_dtypes


base_end_time = time.time()
base_duration = base_end_time - base_start_time
print(f"基本模块导入时间(进入main之前的时间): {base_duration:.4f} 秒")


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
    np_int = numba_config["np"]["int"]
    np_float = numba_config["np"]["float"]
    np_bool = numba_config["np"]["bool"]

    # 在解析参数并更新全局配置字典后，再导入 Numba 函数
    from utils.handle_params import init_params
    from utils.mock_data import get_mock_data

    from parallel_mtf import run_parallel_mtf
    from signals.calculate_signal import SignalId, signal_dict

    # 记录参数生成开始时间，并计算冷启动时间
    if show_timing:
        main_end_time = time.time()
        cold_duration = main_end_time - main_start_time
        print(f"冷启动时间 (进入main到参数生成前): {cold_duration:.4f} 秒")

    if show_timing:
        data_start_time = time.time()
        data_count = 40000
        data_count_mtf = 10000
        tohlcv_np = get_mock_data(data_count=data_count, period="15m")
        tohlcv_np_mtf = get_mock_data(data_count=data_count_mtf, period="4h")
        mapping_mtf = np.zeros(tohlcv_np.shape[0], dtype=np_float)
        data_end_time = time.time()
        data_duration = data_end_time - data_start_time
        print(f"数据导入时间 (进入main到参数生成前): {data_duration:.4f} 秒")

    if show_timing:
        params_start_time = time.time()

    params_count = 2
    signal_select_id = SignalId.signal_3_id.value
    smooth_mode = None
    (
        tohlcv_np,
        indicator_params_list,
        backtest_params_list,
        tohlcv_np_mtf,
        indicator_params_list_mtf,
        mapping_mtf,
        tohlcv_smoothed,
        tohlcv_mtf_smoothed,
    ) = init_params(
        params_count,
        signal_select_id,
        signal_dict,
        tohlcv_np,
        tohlcv_np_mtf,
        mapping_mtf,
        smooth_mode=smooth_mode,
    )

    # 记录参数生成结束时间，并计算运行时间
    if show_timing:
        params_end_time = time.time()
        params_duration = params_end_time - params_start_time
        print(f"参数生成时间 (default_params): {params_duration:.4f} 秒")

    # 记录 parallel_entry 函数开始时间
    if show_timing:
        parallel_start_time = time.time()

    result = run_parallel_mtf(
        tohlcv_np,
        indicator_params_list,
        backtest_params_list,
        tohlcv_np_mtf,
        indicator_params_list_mtf,
        mapping_mtf,
        tohlcv_smoothed,
        tohlcv_mtf_smoothed,
    )
    (
        indicators_output_list,
        signals_output_list,
        backtest_output_list,
        performance_output_list,
        indicators_output_list_mtf,
    ) = result

    # 记录 run_parallel 函数结束时间并打印内核运行时间
    if show_timing:
        parallel_end_time = time.time()
        parallel_duration = parallel_end_time - parallel_start_time
        print(f"run_parallel 内核运行时间: {parallel_duration:.4f} 秒")

    print(f"Numba 函数结果: {len(indicators_output_list)}")

    import pdb

    pdb.set_trace()

    # 记录整个main函数结束时间并打印总运行时间
    if show_timing:
        main_end_time = time.time()
        main_duration = main_end_time - main_start_time
        print(f"main 函数总运行时间: {main_duration:.4f} 秒")


if __name__ == "__main__":
    app.command("main | ma")(main)
    app()
