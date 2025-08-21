import numpy as np
import numba as nb
from numba import njit
from numba.typed import Dict


# --- Numba 配置 ---
cache = False
np_float = np.float64
nb_int = nb.int64


def get_mock_data(data_count, period="15m"):
    """
    生成模拟市场数据（OHLCV），使用NumPy矢量化随机游走。

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

    # 价格随机游走
    price_steps = (np.random.rand(data_count) - 0.5) * 2
    prices = 1000 + np.cumsum(price_steps)

    # 确保价格非负
    prices[prices <= 0] = 1000.0

    # 生成开盘价和收盘价
    open_prices = np.insert(prices[:-1], 0, 1000.0)
    close_prices = prices

    # 生成 high 和 low
    high_prices = (
        np.maximum(open_prices, close_prices) + np.random.rand(data_count) * 0.5
    )
    low_prices = (
        np.minimum(open_prices, close_prices) - np.random.rand(data_count) * 0.5
    )

    # 确保 high >= open, close, low
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    # 确保 low <= open, close, high
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    # 成交量随机游走，并确保非负
    volume_steps = (np.random.rand(data_count) - 0.5) * 500
    volumes = 10000 + np.cumsum(volume_steps)
    volumes[volumes < 100] = 100.0

    # 组合成最终的二维数组
    data = np.vstack(
        [times, open_prices, high_prices, low_prices, close_prices, volumes]
    ).T.astype(np_float)

    return data


# --- 优化后的映射函数 ---
@njit(cache=cache)
def get_data_mapping(tohlcv_np, tohlcv_np_mtf):
    _d = Dict.empty(
        key_type="unicode_type",
        value_type=nb_int[:],
    )
    if (
        tohlcv_np is None
        or tohlcv_np_mtf is None
        or tohlcv_np.shape[0] == 0
        or tohlcv_np_mtf.shape[0] == 0
    ):
        _d["mtf"] = np.zeros(0, dtype=nb_int)
        return _d

    times = tohlcv_np[:, 0]
    mtf_times = tohlcv_np_mtf[:, 0]

    # 使用 np.searchsorted 进行矢量化查找
    # side='right' 找到第一个大于当前时间戳的位置
    mapping_indices = np.searchsorted(mtf_times, times, side="right") - 1

    _d["mtf"] = mapping_indices
    return _d


# --- 示例测试代码 ---
if __name__ == "__main__":
    # 生成模拟数据
    data_15m = get_mock_data(data_count=10, period="15m")
    data_1h = get_mock_data(data_count=3, period="1h")

    # 模拟一个数据缺失
    data_15m_with_gap = np.delete(data_15m, obj=5, axis=0)

    # 测试1: 正常映射
    print("--- 正常映射测试 (15m -> 1h) ---")
    mapping_result_1 = get_data_mapping(data_15m, data_1h)
    print("15m数据时间戳:", data_15m[:, 0])
    print("1h数据时间戳:", data_1h[:, 0])
    print("映射结果 (15m -> 1h):", mapping_result_1["mtf"])

    print("\n--- 验证正常映射 ---")
    for i in range(data_15m.shape[0]):
        mapped_idx = mapping_result_1["mtf"][i]
        if mapped_idx != -1:
            print(
                f"15m time {data_15m[i, 0]} maps to 1h time {data_1h[mapped_idx, 0]} (idx: {mapped_idx})"
            )

    # 测试2: 包含数据缺失的映射
    print("\n--- 包含数据缺失的映射测试 (15m带缺口 -> 1h) ---")
    mapping_result_2 = get_data_mapping(data_15m_with_gap, data_1h)
    print("15m带缺口数据时间戳:", data_15m_with_gap[:, 0])
    print("1h数据时间戳:", data_1h[:, 0])
    print("映射结果 (带缺口):", mapping_result_2["mtf"])

    print("\n--- 验证带缺口映射 ---")
    for i in range(data_15m_with_gap.shape[0]):
        mapped_idx = mapping_result_2["mtf"][i]
        if mapped_idx != -1:
            print(
                f"15m gap time {data_15m_with_gap[i, 0]} maps to 1h time {data_1h[mapped_idx, 0]} (idx: {mapped_idx})"
            )
