import numpy as np
import numba as nb
from numba import njit
from src.utils.constants import numba_config

cache = numba_config["cache"]
nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool_type = numba_config["nb"]["bool"]


# --- PSAR 状态元组定义 ---
PsarState = nb.types.Tuple((nb_bool_type, nb_float, nb_float, nb_float))


# --- PSAR 初始化函数 ---
# 逻辑保持不变，因为它处理的是第一根K线，本身就是标量操作
@njit(
    PsarState(nb_float[:], nb_float[:], nb_float[:], nb_int, nb_float),
    cache=cache,
)
def psar_init(high, low, close, force_direction_int, af0):
    """
    初始化 PSAR 算法的初始状态，可强制指定初始方向。
    直接接收所需的数组数据，取其第一个或最后一个有效值。
    返回一个元组：(is_long, current_psar, current_ep, current_af)
    """
    high_prev = high[0]
    high_curr = high[1]
    low_prev = low[0]
    low_curr = low[1]
    close_prev = close[0]

    is_long = False
    if force_direction_int == 1:  # 强制多头
        is_long = True
    elif force_direction_int == -1:  # 强制空头
        is_long = False
    else:  # 自动判断方向 (force_direction_int == 0)
        up_dm = high_curr - high_prev
        dn_dm = low_prev - low_curr
        is_falling_initial = dn_dm > up_dm and dn_dm > 0
        is_long = not is_falling_initial

    current_psar = close_prev
    current_ep = high_prev if is_long else low_prev
    current_af = af0

    return (is_long, current_psar, current_ep, current_af)


# --- PSAR 第一次迭代函数 ---
# 逻辑保持不变，因为它处理的是第一根K线之后的结果，本身就是标量操作
@njit(
    nb.types.Tuple(
        (
            PsarState,
            nb_float,
            nb_float,
            nb_float,
        )
    )(
        nb_float[:],
        nb_float[:],
        nb_float[:],
        nb_float,
        nb_float,
        nb_float,
    ),
    cache=cache,
)
def psar_first_iteration(high, low, close, af0, af_step, max_af):
    """
    处理 PSAR 算法的第一次迭代（计算索引1的结果）。
    返回一个元组：(new_state_tuple, psar_long_val, psar_short_val, reversal_val)
    """
    initial_state = psar_init(high, low, close, 0, af0)
    is_long, current_psar, current_ep, current_af = initial_state

    # 逻辑与旧代码保持一致，但参数直接来自数组
    if np.isnan(current_psar):
        return (is_long, current_psar, current_ep, current_af), np.nan, np.nan, 0.0

    next_psar_raw_candidate = (
        current_psar + current_af * (current_ep - current_psar)
        if is_long
        else current_psar - current_af * (current_psar - current_ep)
    )
    current_psar = (
        min(next_psar_raw_candidate, low[0])
        if is_long
        else max(next_psar_raw_candidate, high[0])
    )

    reversal = (
        low[1] < next_psar_raw_candidate
        if is_long
        else high[1] > next_psar_raw_candidate
    )

    if is_long:
        if high[1] > current_ep:
            current_ep = high[1]
            current_af = min(max_af, current_af + af_step)
    else:
        if low[1] < current_ep:
            current_ep = low[1]
            current_af = min(max_af, current_af + af_step)

    if reversal:
        is_long = not is_long
        current_af = af0
        current_psar = current_ep
        if is_long:
            if current_psar > low[1]:
                current_psar = low[1]
            current_ep = high[1]
        else:
            if current_psar < high[1]:
                current_psar = high[1]
            current_ep = low[1]

    psar_long_val = current_psar if is_long else np.nan
    psar_short_val = current_psar if not is_long else np.nan
    reversal_val = nb_float(reversal)

    return (
        (is_long, current_psar, current_ep, current_af),
        psar_long_val,
        psar_short_val,
        reversal_val,
    )


# --- PSAR 实时更新函数 ---
# 逻辑保持不变，因为它处理的是单根K线的更新，本身就是标量操作
@njit(
    nb.types.Tuple(
        (
            PsarState,
            nb_float,
            nb_float,
            nb_float,
        )
    )(
        PsarState,
        nb_float,
        nb_float,
        nb_float,
        nb_float,
        nb_float,
        nb_float,
    ),
    cache=cache,
)
def psar_update(
    prev_state,
    current_high,
    current_low,
    prev_high,
    prev_low,
    af_step,
    max_af,
):
    """
    根据前一根K线后的PSAR状态和当前K线的数据，计算新的PSAR值并更新状态。
    返回一个元组：(new_state_tuple, psar_long_val, psar_short_val, reversal)
    """
    prev_is_long, prev_psar, prev_ep, prev_af = prev_state

    # 1. 计算下一根K线的原始 PSAR 候选值
    if prev_is_long:
        next_psar_raw_candidate = prev_psar + prev_af * (prev_ep - prev_psar)
    else:
        next_psar_raw_candidate = prev_psar - prev_af * (prev_psar - prev_ep)

    # 2. 判断是否发生反转
    # 逻辑与旧代码保持一致，但移除了 close_for_reversal 参数，因为它与TA-Lib不一致
    reversal_price = current_low if prev_is_long else current_high
    reversal = nb_float(
        reversal_price < next_psar_raw_candidate
        if prev_is_long
        else reversal_price > next_psar_raw_candidate
    )

    # 3. 对 PSAR 进行穿透检查
    current_psar = (
        min(next_psar_raw_candidate, prev_low)
        if prev_is_long
        else max(next_psar_raw_candidate, prev_high)
    )

    # 4. 更新极端点 (EP) 和加速因子 (AF)
    new_ep = prev_ep
    new_af = prev_af
    if prev_is_long:
        if current_high > new_ep:
            new_ep = current_high
            new_af = min(max_af, prev_af + af_step)
    else:
        if current_low < new_ep:
            new_ep = current_low
            new_af = min(max_af, prev_af + af_step)

    # 5. 处理反转（如果发生）
    new_is_long = prev_is_long
    if reversal == 1.0:
        new_is_long = not prev_is_long
        new_af = af_step
        current_psar = prev_ep
        if new_is_long:
            current_psar = min(current_psar, current_low)
            new_ep = current_high
        else:
            current_psar = max(current_psar, current_high)
            new_ep = current_low

    # 6. 确定返回的 PSAR 值
    psar_long_val = np.nan
    psar_short_val = np.nan
    if new_is_long:
        psar_long_val = current_psar
    else:
        psar_short_val = current_psar

    return (
        (new_is_long, current_psar, new_ep, new_af),
        psar_long_val,
        psar_short_val,
        reversal,
    )


# --- calculate_psar 主计算函数 ---


@njit(
    nb_float[:, :](
        nb_float[:],
        nb_float[:],
        nb_float[:],
        nb_float,
        nb_float,
        nb_float,
    ),
    cache=cache,
)
def calc_psar(high, low, close, af0, af_step, max_af):
    """
    计算PSAR的整个序列，内部创建结果数组并返回。
    """
    n = len(close)
    if n < 2:
        return np.full((0, 4), np.nan, dtype=nb_float)

    # 在函数内部创建结果数组
    psar_results = np.full((n, 4), np.nan, dtype=nb_float)
    psar_long_result = psar_results[:, 0]
    psar_short_result = psar_results[:, 1]
    psar_af_result = psar_results[:, 2]
    psar_reversal_result = psar_results[:, 3]

    # 初始化索引 0 的 af 和 reversal
    psar_af_result[0] = af0
    psar_reversal_result[0] = 0.0

    # 处理索引 1
    (
        initial_state_for_loop,
        psar_long_result[1],
        psar_short_result[1],
        psar_reversal_result[1],
    ) = psar_first_iteration(high, low, close, af0, af_step, max_af)

    is_long, current_psar, current_ep, current_af = initial_state_for_loop
    psar_af_result[1] = current_af

    if np.isnan(current_psar):
        return psar_results

    # 核心循环：从索引 2 开始
    for i in range(2, n):
        prev_state_tuple = (is_long, current_psar, current_ep, current_af)

        (
            new_state_tuple,
            psar_long_val,
            psar_short_val,
            reversal_val,
        ) = psar_update(
            prev_state_tuple,
            high[i],
            low[i],
            high[i - 1],
            low[i - 1],
            af_step,
            max_af,
        )

        is_long, current_psar, current_ep, current_af = new_state_tuple

        psar_long_result[i] = psar_long_val
        psar_short_result[i] = psar_short_val
        psar_af_result[i] = current_af
        psar_reversal_result[i] = reversal_val

    return psar_results
