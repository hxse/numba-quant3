import re


def get_annualization_factor(period: str) -> float:
    """
    根据给定的K线周期（如"1m", "4h", "1d"）计算年化因子。

    参数:
    period (str): K线周期，例如 "1m", "5m", "1h", "4h", "1d", "1w", "1M", "1y"。

    返回:
    float: 对应的年化因子。

    Raises:
    ValueError: 如果周期格式无效或不支持。
    """

    # 修复后的正则表达式，增加了 'y'
    match = re.match(r"(\d+)([mhdMwy])", period)
    if not match:
        raise ValueError(f"不支持的周期格式: {period}")

    value = int(match.group(1))
    unit = match.group(2)

    # 定义不同单位与分钟的转换关系
    unit_to_minutes = {
        "m": 1,
        "h": 60,
        "d": 24 * 60,
        "w": 7 * 24 * 60,
        "M": 30 * 24 * 60,  # 按30天估算
        "y": 365 * 24 * 60,  # 一年
    }

    if unit not in unit_to_minutes:
        raise ValueError(f"不支持的时间单位: {unit}")

    # 计算一年的总分钟数
    minutes_per_year = 365 * 24 * 60

    # 计算当前周期对应的分钟数
    period_in_minutes = value * unit_to_minutes[unit]

    # 计算年化因子
    annualization_factor = minutes_per_year / period_in_minutes

    return annualization_factor
