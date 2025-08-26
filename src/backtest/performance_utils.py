import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict, List
from src.utils.constants import numba_config

from numba import njit, float64
import numpy as np


# 假设 performance_signature 是您定义的类型签名
@njit(cache=True)
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


@njit(cache=True)
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
