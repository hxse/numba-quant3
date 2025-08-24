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

name = "bbands"
input_data_keys = ["close"]
params_config_list = [
    {
        "nb_params": {"period": 14, "std_mult": 2.0},
        "pd_params": {"length": 14, "std": 2.0},
    }
]

# 动态构建 output_key_maps
nb_pd_talib_key_maps = []
for p in params_config_list:
    period = p["pd_params"]["length"]
    std_mult = p["pd_params"]["std"]
    nb_pd_talib_key_maps.append(
        {
            f"{name}_upper": f"BBU_{period}_{std_mult}",
            f"{name}_middle": f"BBM_{period}_{std_mult}",
            f"{name}_lower": f"BBL_{period}_{std_mult}",
            f"{name}_bandwidth": f"BBB_{period}_{std_mult}",
            f"{name}_percent": f"BBP_{period}_{std_mult}",
        }
    )
pd_talib_key_maps = [{v: v for v in d.values()} for d in nb_pd_talib_key_maps]

custom_rtol = 2e-4
assert_func_kwargs = {
    f"{name}_percent": {"custom_rtol": custom_rtol},
    f"BBP_{period}_{std_mult}": {"custom_rtol": custom_rtol},
}


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
