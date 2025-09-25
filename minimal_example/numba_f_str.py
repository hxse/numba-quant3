from numba import njit, types
from numba.typed import Dict, List
import numpy as np


@njit
def test_fstring_with_list_key(nums):
    my_dict = Dict.empty(key_type=types.unicode_type, value_type=types.int64)

    for i in nums:
        key = f"key_{i}"
        my_dict[key] = 100 + i

    for k, v in enumerate(nums):
        key = f"key_{k}"
        my_dict[key] = 200 + v

    value = 10
    my_dict[f"key_{value}"] = 300

    return my_dict


# 创建一个 Numba 的列表
nums_list = List.empty_list(types.int64)
nums_list.append(5)
nums_list.append(6)
nums_list.append(7)

print("尝试运行测试用例...")
try:
    res = test_fstring_with_list_key(nums_list)
    print(res)
except Exception as e:
    print("\n--- 编译失败 ---")
    print("错误类型:", type(e))
    print("错误信息:", e)
