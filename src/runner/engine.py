from src.utils.common import time_it, assert_attr_is_not_none


class Engine:
    def _run_parallel_backtest(self):
        """
        执行并行回测并返回结果。
        """

        assert_attr_is_not_none(self, "params_tuple")

        self.result_tuple = self.run_parallel(*self.params_tuple)
