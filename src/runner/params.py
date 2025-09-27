from src.utils.common import time_it, assert_attr_is_not_none


class ParamsInitializer:
    def _initialize_backtest_params(self):
        """
        初始化回测参数。
        """

        assert_attr_is_not_none(
            self,
            "select_id",
            "params_count",
            "is_only_performance",
            "tohlcv_np_list",
            "period_list",
            "smooth_mode",
            "use_presets_indicator_params",
            "use_presets_backtest_params",
        )

        signal_select_id = self.SignalId[self.select_id].value

        if isinstance(self.is_only_performance, str):
            _ = self.params_count != 1
        else:
            assert isinstance(self.is_only_performance, bool), (
                "is_only_performance需要str或bool"
            )
            _ = self.is_only_performance
        self.is_only_performance = _

        self.params_tuple = self.init_params(
            self.params_count,
            signal_select_id,
            self.signal_dict,
            ohlcv_mtf_np_list=self.tohlcv_np_list,
            period_list=self.period_list,
            smooth_mode=self.smooth_mode,
            is_only_performance=self.is_only_performance,
            use_presets_indicator_params=self.use_presets_indicator_params,
            use_presets_backtest_params=self.use_presets_backtest_params,
        )
