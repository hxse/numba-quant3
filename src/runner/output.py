from src.utils.common import time_it, assert_attr_is_not_none


class OutputProcessor:
    def _process_backtest_output(self):
        """
        处理和转换并行回测的输出。
        """
        assert_attr_is_not_none(
            self,
            "params_tuple",
            "result_tuple",
            "data_suffix",
            "params_suffix",
            "convert_num",
        )

        # 转换数据
        self.result_converted, self.data_list = self.process_data_output(
            params=self.params_tuple,
            data_list=self.result_tuple,
            convert_num=self.convert_num,
            data_suffix=self.data_suffix,
            params_suffix=self.params_suffix,
        )
