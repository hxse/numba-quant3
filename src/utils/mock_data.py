import numpy as np


# 从全局配置中获取 Numba 类型
from src.utils.constants import numba_config


np_float = numba_config["np"]["float"]


def get_mock_data(data_count, period="15m"):
    """
    生成模拟市场数据（OHLCV），使用几何布朗运动和动态成交量。

    参数:
        data_count (int): 要生成的K线数量。
        period (str): K线的周期，例如 "3m", "5m", "15m", "1h", "4h", "1d" 等。

    返回:
        np.ndarray: 一个二维NumPy数组，包含 timestamp, o, h, l, c, v。
    """
    np.random.seed(42)

    # 将周期字符串映射到秒数
    period_to_seconds = {
        "3m": 3 * 60,
        "5m": 5 * 60,
        "15m": 15 * 60,
        "30m": 30 * 60,
        "1h": 60 * 60,
        "2h": 2 * 60 * 60,
        "4h": 4 * 60 * 60,
        "6h": 6 * 60 * 60,
        "12h": 12 * 60 * 60,
        "1d": 24 * 60 * 60,
        "1w": 7 * 24 * 60 * 60,
    }

    if period not in period_to_seconds:
        raise ValueError(
            f"Unsupported period: {period}. Supported periods are {list(period_to_seconds.keys())}"
        )

    time_step_seconds = period_to_seconds[period]

    # 初始化时间序列，使用毫秒级时间戳
    start_time_ms = 1677600000000  # 沿用参考CSV中的毫秒级时间戳
    time_step_ms = time_step_seconds * 1000
    times = np.arange(
        start_time_ms,
        start_time_ms + data_count * time_step_ms,
        time_step_ms,
        dtype=np_float,
    )

    # 几何布朗运动参数
    initial_price = 1000.0  # 初始价格
    mu = 0.0001  # 漂移系数（年化），模拟长期趋势
    sigma = 0.02  # 波动率（年化），控制价格波动幅度
    dt = time_step_seconds / (365 * 24 * 60 * 60)  # 时间步长（年化）

    # 生成对数价格路径
    log_returns = np.random.normal(
        loc=(mu - 0.5 * sigma**2) * dt, scale=sigma * np.sqrt(dt), size=data_count
    )
    log_prices = np.log(initial_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    # 确保价格非负
    prices[prices <= 0] = initial_price

    # 生成开盘价和收盘价
    open_prices = np.insert(prices[:-1], 0, initial_price)
    close_prices = prices

    # 生成高低价，波动幅度与周期相关
    period_scale = np.sqrt(time_step_seconds / (15 * 60))  # 以15分钟为基准缩放波动
    high_prices = np.maximum(open_prices, close_prices) * (
        1 + np.random.lognormal(mean=0, sigma=0.01 * period_scale, size=data_count)
    )
    low_prices = np.minimum(open_prices, close_prices) * (
        1 - np.random.lognormal(mean=0, sigma=0.01 * period_scale, size=data_count)
    )

    # 确保 high >= open, close, low
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    # 生成成交量，与价格波动相关
    price_changes = np.abs(np.diff(prices, prepend=initial_price)) / prices
    base_volume = 10000.0
    volume_volatility = 500.0 * period_scale  # 成交量波动随周期放大
    volumes = base_volume + volume_volatility * price_changes * np.random.lognormal(
        mean=0, sigma=0.5, size=data_count
    )
    volumes[volumes < 100] = 100.0

    # 组合成最终的二维数组
    data = np.vstack(
        [times, open_prices, high_prices, low_prices, close_prices, volumes]
    ).T.astype(np_float)

    return data
