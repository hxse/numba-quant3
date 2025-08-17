import numpy as np


# 从全局配置中获取 Numba 类型
from src.utils.constants import numba_config


np_float = numba_config["np"]["float"]


def get_mock_data(data_count):
    """
    生成模拟市场数据（OHLCV），使用NumPy矢量化随机游走。

    参数:
        data_count (int): 要生成的K线数量。

    返回:
        np.ndarray: 一个二维NumPy数组，包含time, o, h, l, c, v。
    """
    np.random.seed(42)

    # 初始化时间序列
    start_time = 1672531200
    time_step = 60
    times = np.arange(
        start_time, start_time + data_count * time_step, time_step, dtype=np_float
    )

    # 价格随机游走
    price_steps = (np.random.rand(data_count) - 0.5) * 2
    prices = 1000 + np.cumsum(price_steps)

    # 确保价格非负
    prices[prices <= 0] = 1000.0  # 或者使用更复杂的逻辑

    # 生成开盘价和收盘价
    open_prices = np.insert(prices[:-1], 0, 1000.0)
    close_prices = prices

    # 生成 high 和 low
    # 这里的逻辑是简化版的，为了保持矢量化
    high_prices = (
        np.maximum(open_prices, close_prices) + np.random.rand(data_count) * 0.5
    )
    low_prices = (
        np.minimum(open_prices, close_prices) - np.random.rand(data_count) * 0.5
    )

    # 成交量随机游走，并确保非负
    volume_steps = (np.random.rand(data_count) - 0.5) * 500
    volumes = 10000 + np.cumsum(volume_steps)
    volumes[volumes < 100] = 100.0

    # 组合成最终的二维数组
    data = np.vstack(
        [times, open_prices, high_prices, low_prices, close_prices, volumes]
    ).T.astype(np_float)

    return data
