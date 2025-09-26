from .data import DataLoader
from .params import ParamsInitializer
from .engine import Engine
from .output import OutputProcessor
from .archive import DataArchiver


from src.utils.common import time_it, assert_attr_is_not_none


class BacktestRunner(
    DataLoader, ParamsInitializer, Engine, OutputProcessor, DataArchiver
):
    def __init__(
        self, enable_cache: bool, enable64: bool, enable_warmup: bool, show_timing: bool
    ):
        """
        初始化回测运行器，设置参数并执行初始配置。

        Args:
            enable_cache (bool): 是否启用 Numba 缓存。
            enable64 (bool): 是否启用 64 位浮点数。
            enable_warmup (bool): 是否启用预运行。
        """
        self.enable_cache = enable_cache
        self.enable64 = enable64
        self.enable_warmup = enable_warmup
        self.show_timing = show_timing
        # 初始化所有实例属性为 None
        self.symbol = None
        self.period_list = None
        self.data_count_list = None
        self.select_id = None
        self.params_count = None
        self.convert_num = None
        self.tohlcv_np_list = None
        self.params_tuple = None
        self.is_only_performance = None
        self.result_tuple = None
        self.result_converted = None
        self.data_list = None
        self.smooth_mode = None
        self.data_path = None
        self.data_suffix = None
        self.params_suffix = None

        with time_it(self.show_timing, "导入numba模块时间"):
            self._setup_numba_config()
            self._import_parallel_modules()

    def _setup_numba_config(self):
        """
        根据命令行参数设置 Numba 的全局配置。
        """
        from src.utils.constants import numba_config, set_numba_dtypes

        set_numba_dtypes(
            numba_config, enable64=self.enable64, enable_cache=self.enable_cache
        )
        self.enable_cache = numba_config["enable_cache"]

    def _import_parallel_modules(self):
        """
        导入与并行计算相关的模块。
        """
        import numpy as np
        from src.utils.mock_data import get_mock_data
        from src.convert_params.param_initializer import init_params
        from src.parallel import run_parallel
        from src.convert_output.process_data import process_data_output
        from src.convert_output.archive_manager import archive_data
        from src.convert_output.server_upload import get_token, get_local_dir
        from src.signals.calculate_signal import SignalId, signal_dict

        self.np = np
        self.get_mock_data = get_mock_data
        self.init_params = init_params
        self.run_parallel = run_parallel
        self.process_data_output = process_data_output
        self.archive_data = archive_data
        self.get_token = get_token
        self.get_local_dir = get_local_dir
        self.SignalId = SignalId
        self.signal_dict = signal_dict

    def run(
        self,
        #
        symbol: str = "mock",
        period_list: list[str] = ["15m", "1h"],
        data_count_list: list[int] = [10000, 5000],
        params_count: int = 1,
        select_id: str = "signal_3_id",
        convert_num: int = 0,
        smooth_mode: str = "",
        is_only_performance: bool | str = "",  # 如果是str则视为auto模式
        #
        data_path="./data",
        data_suffix=".csv",
        params_suffix=".json",
    ):
        """
        执行完整的业务逻辑流程，包括数据加载、回测和数据存档。
        """
        # 将所有参数设置到实例属性中，供其他私有方法使用
        self.symbol = symbol
        self.period_list = period_list
        self.data_count_list = data_count_list
        self.params_count = params_count
        self.select_id = select_id
        self.convert_num = convert_num
        self.smooth_mode = smooth_mode
        self.is_only_performance = is_only_performance
        #
        self.data_path = data_path
        self.data_suffix = data_suffix
        self.params_suffix = params_suffix

        # 循环执行回测，第一次为预热
        for i in range(2):
            with time_it(
                self.show_timing,
                "回测热身" if i == 0 else "回测流程",
            ):
                if i == 0 and not self.enable_warmup:
                    continue

                if i == 0:
                    self.params_count = 1
                    self.symbol = "mock"
                    self.period_list = ["15m", "1h"]
                    self.data_count_list = [100, 50]
                    self.select_id = "signal_3_id"
                    self.convert_num = 0
                    self.smooth_mode = ""
                    self.is_only_performance = False
                else:
                    self.params_count = params_count
                    self.symbol = symbol
                    self.period_list = period_list
                    self.data_count_list = data_count_list
                    self.select_id = select_id
                    self.convert_num = convert_num
                    self.smooth_mode = smooth_mode
                    self.is_only_performance = is_only_performance

                with time_it(self.show_timing and i > 0, "数据导入"):
                    self._load_data()

                with time_it(self.show_timing and i > 0, "创建参数"):
                    self._initialize_backtest_params()

                with time_it(self.show_timing and i > 0, "核心回测"):
                    self._run_parallel_backtest()

                with time_it(self.show_timing and i > 0, "转换输出"):
                    self._process_backtest_output()

        with time_it(self.show_timing, "数据存档"):
            self._archive_data()

        return self.result_converted, self.data_list
