import numpy as np
import numba as nb
from numba import njit, prange

from src.utils.constants import numba_config


cache = numba_config["cache"]

nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]

print("run_entery cache", cache)


from parallel import run_parallel
from parallel_mtf_A import run_parallel_mtf_A


@njit(cache=cache, parallel=True)
def run_parallel_mtf_B(
    tohlcv_np,
    indicator_params_list,
    backtest_params_list,
    tohlcv_np_mtf=None,
    indicator_params_list_mtf=None,
    mapping_mtf=None,
):
    """
    # @njit(cache=cache) 这个不能这样用,有bug,参考https://github.com/numba/numba/issues/10184
    并发200配置和4万数据,如果加上njit,缓存,parallel,这个是0.1355 秒,0.1460 秒,0.1437 秒
    并发200配置和4万数据,如果去掉njit,缓存,parallel,这个是0.2306 秒,0.2024 秒,0.2189 秒
    并发1  配置和4万数据,如果加上njit,缓存,parallel,这个是0.0606 秒,0.0574 秒,0.0565 秒
    """
    if mapping_mtf is not None:
        assert tohlcv_np.shape[0] == len(mapping_mtf), "mapping数据应该和tohlcv数据相等"

    _indicators_output_list = None

    for i in prange(0):  # 因为用了parallel,所以用prange避免警告
        _i = i

    if (
        tohlcv_np_mtf is not None
        and indicator_params_list_mtf is not None
        and mapping_mtf is not None
    ):
        (
            _indicators_output_list,
            _signals_output_list,
            _backtest_output_list,
            _performance_output_list,
            _indicators_output_list_large,
        ) = run_parallel(
            tohlcv_np_mtf,
            indicator_params_list_mtf,
            backtest_params_list,
            indicators_output_list_large=None,
            mapping_large=None,
        )

    (
        indicators_output_list,
        signals_output_list,
        backtest_output_list,
        performance_output_list,
        indicators_output_list_large,
    ) = run_parallel(
        tohlcv_np,
        indicator_params_list,
        backtest_params_list,
        indicators_output_list_large=_indicators_output_list,
        mapping_large=mapping_mtf,
    )

    return (
        indicators_output_list,
        signals_output_list,
        backtest_output_list,
        performance_output_list,
        indicators_output_list_large,
    )
