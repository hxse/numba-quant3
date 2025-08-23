import sys
from pathlib import Path

root_path = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()),
    None,
)
if root_path:
    sys.path.insert(0, str(root_path))

from Test.utils.over_constants import numba_config


import numpy as np


from src.utils.nb_params import (
    create_params_list_template,
    create_params_dict_template,
    get_params_list_value,
    set_params_list_value,
    get_params_dict_value,
    set_params_dict_value,
    convert_params_dict_list,
)

np_float = numba_config["np"]["float"]


def test_params_config():
    # 测试基于列表的参数配置
    (
        indicator_params_as_list,
        backtest_params_as_list,
    ) = create_params_list_template(params_count=2, empty=False)

    initial_list_as_arr = get_params_list_value("sma_period", indicator_params_as_list)
    assert initial_list_as_arr.shape == (2,)
    assert np.allclose(initial_list_as_arr, np.array([14.0, 14.0], dtype=np_float))

    set_params_list_value(
        "sma_period", indicator_params_as_list, np.array([1, 2], dtype=np_float)
    )
    assert np.allclose(indicator_params_as_list[0]["sma_period"], 1.0)
    assert np.allclose(indicator_params_as_list[1]["sma_period"], 2.0)

    # 测试基于字典的参数配置
    (
        indicator_params_as_dict,
        backtest_params_as_dict,
    ) = create_params_dict_template(params_count=2, empty=False)

    # 从字典形式转换为列表形式
    indicator_list_from_dict = convert_params_dict_list(indicator_params_as_dict)
    assert len(indicator_list_from_dict) == 2
    assert np.allclose(indicator_list_from_dict[0]["sma_period"], 14)
    assert np.allclose(indicator_list_from_dict[1]["sma_period"], 14)

    # 获取和设置字典中的值
    initial_dict_as_arr = get_params_dict_value("sma_period", indicator_params_as_dict)
    assert np.allclose(initial_dict_as_arr, np.array([14.0, 14.0], dtype=np_float))

    set_params_dict_value(
        "sma_period", indicator_params_as_dict, np.array([1, 2], dtype=np_float)
    )
    assert np.allclose(
        indicator_params_as_dict["sma_period"], np.array([1.0, 2.0], dtype=np_float)
    )
