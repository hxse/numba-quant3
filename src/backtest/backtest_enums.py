import numpy as np
from numba import njit
from numba.core import types

from src.utils.constants import numba_config

cache = numba_config["cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


from enum import IntEnum, unique


@unique
class PositionStatus(IntEnum):
    """
    定义回测中的仓位状态，使用整数值方便 Numba 处理。
    """

    NO_POSITION = 0  # 无仓位
    ENTER_LONG = 1  # 开多
    HOLD_LONG = 2  # 持多
    EXIT_LONG = 3  # 平多
    REVERSE_TO_LONG = 4  # 平空开多
    ENTER_SHORT = -1  # 开空
    HOLD_SHORT = -2  # 持空
    EXIT_SHORT = -3  # 平空
    REVERSE_TO_SHORT = -4  # 平多开空


ps = PositionStatus


# 新增：Numba 兼容的辅助函数
@njit(cache=cache)
def is_long_position(status_int):
    return status_int in (
        ps.ENTER_LONG.value,
        ps.HOLD_LONG.value,
        ps.REVERSE_TO_LONG.value,
    )


@njit(cache=cache)
def is_short_position(status_int):
    return status_int in (
        ps.ENTER_SHORT.value,
        ps.HOLD_SHORT.value,
        ps.REVERSE_TO_SHORT.value,
    )


@njit(cache=cache)
def is_no_position(status_int):
    return status_int in (
        ps.NO_POSITION.value,
        ps.EXIT_LONG.value,
        ps.EXIT_SHORT.value,
    )
