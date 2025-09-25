import numpy as np
from numba import types, njit
from numba.typed import Dict, List
from numba.extending import overload
import time

# 模拟您的配置文件
numba_config = {"enable_cache": False}
enable_cache = numba_config["enable_cache"]
nb_int = types.int64
nb_float = types.float64
nb_bool = types.boolean
nb_str = types.unicode_type


# 这是我们要重载的 Python 函数
@njit(cache=enable_cache)
def get_dict_keys_and_values_wrapper(params_dict):
    """一个占位符函数，用于 Numba 重载"""
    return get_dict_keys_and_values(params_dict)


# 这是我们要重载的 Python 函数
def get_dict_keys_and_values(params_dict):
    """一个占位符函数，用于 Numba 重载"""
    pass


# 这是我们重载的实现
@overload(get_dict_keys_and_values, jit_options={"cache": enable_cache})
def ov_get_dict_keys_and_values(params_dict):
    """
    Numba 重载函数，利用闭包动态生成实现。
    """
    # 1. 检查输入是否为 Numba Dict 类型对象
    assert isinstance(params_dict, types.DictType), (
        f"需要字典类型, 实际为{type(params_dict)}"
    )

    # 2. 从类型对象中获取键和值类型
    key_type = params_dict.key_type
    value_type = params_dict.value_type

    # 4. 返回一个纯 Python 函数作为实现
    # 确保参数名称与 ov_get_dict_keys_and_values 一致
    def impl(params_dict):
        # 这里的 params_dict 是一个实际的 Numba Dict 实例
        keys = List.empty_list(key_type)
        values = List.empty_list(value_type)

        for key, value in params_dict.items():
            keys.append(key)
            values.append(value)

        return (keys, values)

    # 5. 返回 impl 函数。
    return impl


if __name__ == "__main__":
    # --- 测试部分 ---

    for i in range(3):
        start_time = time.perf_counter()

        # 1. 测试整型标量值
        print("--- 测试整型标量 ---")
        d1 = Dict.empty(nb_str, nb_int)
        d1["a"] = 1
        d1["b"] = 2
        d1["c"] = 3
        keys1, values1 = get_dict_keys_and_values_wrapper(d1)
        print(f"键列表: {keys1}")
        print(f"值列表: {values1}")
        print("-" * 20)

        # 2. 测试浮点型数组值
        print("--- 测试浮点型数组 ---")
        d2 = Dict.empty(nb_str, nb_float[:])
        d2["x"] = np.array([1.1, 2.2], dtype=np.float64)
        d2["y"] = np.array([3.3, 4.4], dtype=np.float64)
        keys2, values2 = get_dict_keys_and_values_wrapper(d2)
        print(f"键列表: {keys2}")
        print(f"值列表: {values2}")
        print("-" * 20)

        # 3. 测试布尔型标量值
        print("--- 测试布尔型标量 ---")
        d3 = Dict.empty(nb_str, nb_bool)
        d3["is_true"] = True
        d3["is_false"] = False
        keys3, values3 = get_dict_keys_and_values_wrapper(d3)
        print(f"键列表: {keys3}")
        print(f"值列表: {values3}")
        print("-" * 20)

        print(f"第{i + 1}次运行时间 {time.perf_counter() - start_time}")
