import time

base_start_time = time.perf_counter()

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

from memory_profiler import profile


def main(
    cache: bool = typer.Option(True, "--cache/--no-cache", help="启用或禁用Numba缓存"),
    enable64: bool = typer.Option(True, "--64bit/--32bit", help="启用或禁用64位浮点数"),
    show_timing: bool = typer.Option(
        True, "--timing/--no-timing", help="启用或禁用运行时间打印。"
    ),
    enable_profile: bool = typer.Option(
        False, "--profile/--no-profile", help="启用或禁用内存分析器profile装饰器"
    ),
    enable_warmup: bool = typer.Option(
        False, "--warmup/--no-warmup", help="启用或禁用提前预运行一次"
    ),
):
    """
    执行整个回测流程并可选地打印运行时间。
    """
    params = (cache, enable64, show_timing, enable_warmup)
    if enable_profile:
        _func = profile(run_main_logic)
        _func(*params)
    else:
        run_main_logic(*params)


def run_main_logic(cache, enable64, show_timing, enable_warmup):
    if show_timing:
        base_end_time = time.perf_counter()
        base_duration = base_end_time - base_start_time
        print(f"基本模块导入时间: {base_duration:.4f} 秒")

    if show_timing:
        main_start_time = time.perf_counter()

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
    from utils.convert_output import convert_output, archive_data

    from parallel import run_parallel
    from signals.calculate_signal import SignalId, signal_dict

    if show_timing:
        cold_duration = time.perf_counter() - main_start_time
        print(f"numba模块导入时间: {cold_duration:.4f} 秒")

    if show_timing:
        data_start_time = time.perf_counter()

    symbol = "mock"
    data_count = 40000
    data_count_mtf = 10000
    period = "15m"
    period_mtf = "4h"
    tohlcv_np = get_mock_data(data_count=data_count, period=period)
    tohlcv_np_mtf = get_mock_data(data_count=data_count_mtf, period=period_mtf)
    mapping_mtf = np.zeros(tohlcv_np.shape[0], dtype=np_float)

    if show_timing:
        data_end_time = time.perf_counter()
        data_duration = data_end_time - data_start_time
        print(f"数据导入时间: {data_duration:.4f} 秒")

    for i in range(2):
        if show_timing:
            params_start_time = time.perf_counter()

        if i == 0 and not enable_warmup:
            continue

        params_count = 1 if i == 0 else 1
        signal_select_id = SignalId.signal_3_id.value
        smooth_mode = None
        is_only_performance = False if params_count == 1 else True
        print("k线数量", len(tohlcv_np), "并发数量", params_count)

        params_tuple = init_params(
            params_count,
            signal_select_id,
            signal_dict,
            tohlcv_np,
            tohlcv_np_mtf=tohlcv_np_mtf,
            mapping_mtf=mapping_mtf,
            smooth_mode=smooth_mode,
            period=period,
            is_only_performance=is_only_performance,
        )
        (
            tohlcv,
            indicator_params_list,
            backtest_params_list,
            tohlcv_mtf,
            indicator_params_list_mtf,
            mapping_mtf,
            tohlcv_smoothed,
            tohlcv_mtf_smoothed,
            is_only_performance,
        ) = params_tuple

        if show_timing:
            params_end_time = time.perf_counter()
            params_duration = params_end_time - params_start_time
            print(f"默认参数生成运行时间: {params_duration:.4f} 秒")

        if show_timing:
            parallel_start_time = time.perf_counter()

        result_tuple = run_parallel(*params_tuple)
        (
            indicators_output_list,
            signals_output_list,
            backtest_output_list,
            performance_output_list,
            indicators_output_list_mtf,
        ) = result_tuple

        if show_timing:
            parallel_end_time = time.perf_counter()
            parallel_duration = parallel_end_time - parallel_start_time
            print(f"run_parallel 内核运行时间: {parallel_duration:.4f} 秒")

    if show_timing:
        convert_start_time = time.perf_counter()

    num = 0

    # 转换数据
    final_result, data_list = convert_output(
        params_tuple,
        result_tuple,
        num=num,
        data_suffix=".csv",
    )
    print(f"num {num} final_result {final_result['performance']['total_profit_pct']}")

    if show_timing:
        convert_end_time = time.perf_counter()
        convert_duration = convert_end_time - convert_start_time
        print(f"转换输出运行时间: {convert_duration:.4f} 秒")

    if show_timing:
        archive_start_time = time.perf_counter()

    root_path = "./data"
    token_path = Path(f"{root_path}/config.json")
    child_path = f"{symbol}/{signal_select_id}"
    output_path = f"{root_path}/output/{child_path}"
    # 存档数据
    archive_data(
        data_list,
        # save_local_dir=output_path,
        # save_zip_dir=output_path,
        upload_server="http://127.0.0.1:5123/file/upload",
        server_dir=child_path,
        token_path=token_path,
    )

    if show_timing:
        archive_end_time = time.perf_counter()
        archive_duration = archive_end_time - archive_start_time
        print(f"数据存档运行时间: {archive_duration:.4f} 秒")

    # 记录整个main函数结束时间并打印总运行时间
    if show_timing:
        main_duration = time.perf_counter() - main_start_time
        print(f"main 函数总运行时间: {main_duration:.4f} 秒")


if __name__ == "__main__":
    app.command("main | ma")(main)
    app()
