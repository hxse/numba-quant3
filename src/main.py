import sys
from pathlib import Path
from memory_profiler import profile

# 在解析命令行参数之前导入通用配置
root_path = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()),
    None,
)
if root_path:
    sys.path.insert(0, str(root_path))
# 在此不导入 Numba 相关的模块，为了延迟导入以配置缓存和计时

# 导入 Typer 和应用程序对象
from src.utils.typer_tool import typer, app  # noqa: E402

from src.runner.base import BacktestRunner  # noqa: E402


@app.command("main | ma")
def main(
    data_path: str = typer.Option("./data", help="设置data目录"),
    enable_cache: bool = typer.Option(
        True, "--cache/--no-cache", help="启用或禁用Numba缓存"
    ),
    enable64: bool = typer.Option(True, "--64bit/--32bit", help="启用或禁用64位浮点数"),
    show_timing: bool = typer.Option(
        True, "--timing/--no-timing", help="启用或禁用运行时间打印。"
    ),
    enable_profile: bool = typer.Option(
        False, "--profile/--no-profile", help="启用或禁用内存分析器profile装饰器"
    ),
    enable_warmup: bool = typer.Option(
        True, "--warmup/--no-warmup", help="启用或禁用提前预运行一次"
    ),
):
    """
    执行整个回测流程并可选地打印运行时间。
    """

    params = {
        "symbol": "mock",
        "period_list": ["15m", "1h"],
        "data_count_list": [40000, 10000],
        "params_count": 1,
        "select_id": "signal_3_id",
        #
        "data_path": "./data",
        "data_suffix": ".csv",
        # "data_suffix": ".parquet",
        "params_suffix": ".json",
    }

    runner = BacktestRunner(
        enable_cache=enable_cache,
        enable64=enable64,
        enable_warmup=enable_warmup,
        show_timing=show_timing,
    )

    if enable_profile:
        # 使用 profile 装饰器包装 run 方法
        profiled_run = profile(runner.run)
        result = profiled_run(**params)
    else:
        result = runner.run(**params)

    result_converted, data_list = result
    # import pdb

    # pdb.set_trace()
    return result


if __name__ == "__main__":
    app()
