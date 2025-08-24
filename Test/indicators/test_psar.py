import sys
from pathlib import Path

root_path = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()),
    None,
)
if root_path:
    sys.path.insert(0, str(root_path))


from Test.utils.over_constants import numba_config


from Test.indicators.indicators_template import (
    compare_indicator_accuracy,
    compare_pandas_ta_with_talib,
)


from Test.utils.conftest import np_data_mock, df_data_mock


np_float = numba_config["np"]["float"]

name = "psar"
input_data_keys = ["high", "low", "close"]
params_config_list = [
    {
        "nb_params": {"af0": 0.02, "af_step": 0.02, "max_af": 0.2},
        "pd_params": {"af0": 0.02, "af": 0.02, "max_af": 0.2},
    }
]

# 动态构建 output_key_maps
nb_pd_talib_key_maps = []
for p in params_config_list:
    af0 = p["pd_params"]["af0"]
    max_af = p["pd_params"]["max_af"]
    nb_pd_talib_key_maps.append(
        {
            f"{name}_long": f"PSARl_{af0}_{max_af}",
            f"{name}_short": f"PSARs_{af0}_{max_af}",
            f"{name}_af": f"PSARaf_{af0}_{max_af}",
            f"{name}_reversal": f"PSARr_{af0}_{max_af}",
        }
    )
pd_talib_key_maps = [{v: v for v in d.values()} for d in nb_pd_talib_key_maps]
assert_func_kwargs = {}


def test_accuracy(
    np_data_mock,
    df_data_mock,
    talib=False,
    assert_mode=True,
):
    compare_indicator_accuracy(
        name=name,
        params_config_list=params_config_list,
        tohlcv_np=np_data_mock,
        df_data_mock=df_data_mock,
        input_data_keys=input_data_keys,
        talib=talib,
        assert_mode=assert_mode,
        output_key_maps=nb_pd_talib_key_maps,
        assert_func_kwargs=assert_func_kwargs,
    )


def test_accuracy_talib(np_data_mock, df_data_mock, talib=True, assert_mode=True):
    test_accuracy(np_data_mock, df_data_mock, talib=talib, assert_mode=assert_mode)


def test_pandas_ta_and_talib(df_data_mock, assert_mode=True):
    compare_pandas_ta_with_talib(
        name=name,
        params_config_list=params_config_list,
        df_data_mock=df_data_mock,
        input_data_keys=input_data_keys,
        assert_mode=assert_mode,
        assert_func_kwargs=assert_func_kwargs,
        output_key_maps=pd_talib_key_maps,
    )
