from src.utils.common import time_it, assert_attr_is_not_none


class DataLoader:
    def _load_data(self):
        """
        加载模拟数据。
        """
        assert_attr_is_not_none(self, "period_list", "data_count_list")
        assert isinstance(self.period_list, list), "period应该是list"
        assert isinstance(self.data_count_list, list), "data_count应该是list"
        assert len(self.period_list) > 0 and len(self.data_count_list) > 0, (
            "period和data_count长度要求至少为1"
        )
        assert len(self.period_list) == len(self.data_count_list), (
            "period和data_count长度需要相等"
        )

        self.tohlcv_np_list = [
            self.get_mock_data(data_count=d, period=p)
            for d, p in zip(self.data_count_list, self.period_list)
        ]
