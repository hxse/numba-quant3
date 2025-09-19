import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict, List
from src.utils.constants import numba_config

from numba import njit, float64
import numpy as np


enable_cache = numba_config["enable_cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


# 假设 performance_signature 是您定义的类型签名
@njit(cache=enable_cache)
def calc_sharpe(equity, annualization_factor):
    if len(equity) < 2 or annualization_factor <= 0:
        return 0.0

    # 矢量化计算每根K线的收益率
    returns = (equity[1:] - equity[:-1]) / equity[:-1]

    # 计算平均收益和标准差
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    # 计算夏普比率
    if std_return > 0:
        # 年化夏普比率公式
        sharpe_ratio = (mean_return * annualization_factor) / (
            std_return * np.sqrt(annualization_factor)
        )
    else:
        sharpe_ratio = 0.0

    return sharpe_ratio


@njit(cache=enable_cache)
def calc_calmar(equity, drawdown, annualization_factor):
    if len(equity) < 2 or annualization_factor <= 0 or equity[0] == 0:
        return 0.0

    total_bars = len(equity)

    # 1. 计算年化收益率
    total_years = total_bars / annualization_factor
    # 使用最后和最初的净值计算总收益
    annual_return = (equity[-1] / equity[0]) ** (1 / total_years) - 1

    # 2. 计算最大回撤
    max_drawdown = np.max(drawdown)

    # 3. 计算卡尔马比率
    if max_drawdown > 0:
        calmar_ratio = annual_return / max_drawdown
    else:
        # 如果没有回撤，比率趋向于无穷大
        calmar_ratio = np.inf

    return calmar_ratio


@njit(cache=enable_cache)
def calc_sortino(equity, annualization_factor, min_acceptable_return):
    if len(equity) < 2 or annualization_factor <= 0:
        return 0.0

    # 1. 计算每根K线的收益率
    returns = (equity[1:] - equity[:-1]) / equity[:-1]

    # 2. 筛选出低于最小可接受收益率的“下行”收益
    downside_returns = returns[returns < min_acceptable_return]

    # 3. 计算下行标准差 (Downside Deviation)
    if len(downside_returns) > 0:
        # 下行标准差的计算逻辑
        downside_deviation = np.sqrt(
            np.mean((downside_returns - min_acceptable_return) ** 2)
        )
    else:
        # 如果没有下行收益，下行标准差为0
        downside_deviation = 0.0

    # 4. 计算年化收益率
    mean_return = np.mean(returns)

    # 5. 计算索提诺比率
    if downside_deviation > 0:
        # 年化公式: 索提诺比率 = (年化平均收益 - 年化无风险利率) / 年化下行标准差
        # 这里为了简化，假设无风险利率为0，所以分子只用年化平均收益
        sortino_ratio = (mean_return * annualization_factor) / (
            downside_deviation * np.sqrt(annualization_factor)
        )
    else:
        # 如果没有下行波动，索提诺比率趋向于无穷大
        sortino_ratio = np.inf if mean_return > 0 else 0.0

    return sortino_ratio
