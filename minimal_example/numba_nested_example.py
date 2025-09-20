import numpy as np
from numba import njit, types, typeof
from numba.typed import Dict, List


def run_and_verify(func, name):
    """一个简单的验证函数，用于打印并检查结果。"""
    print(f"--- 验证: {name} ---")
    try:
        result = func()
        print("✅ 成功创建并返回:")
        print(result)
        print(f"   返回的 Numba 类型: {typeof(result)}")
        print("\n")
    except Exception as e:
        print(f"❌ 失败: {e}")
        print("\n")


# 定义 Numba 类型别名
nb_float = types.float64
nb_bool = types.boolean


# 定义一个新的 Numba 函数，用于测试 list 嵌套 list
@njit
def create_nested_list_of_lists():
    # 创建一个包含列表的列表
    outer_list = List.empty_list(List.empty_list(types.int64))

    # 创建并填充内部列表
    inner_list_1 = List.empty_list(types.int64)
    inner_list_1.append(1)
    inner_list_1.append(2)

    inner_list_2 = List.empty_list(types.int64)
    inner_list_2.append(3)
    inner_list_2.append(4)

    # 将内部列表添加到外部列表中
    outer_list.append(inner_list_1)
    outer_list.append(inner_list_2)

    return outer_list


# 定义一个新的 Numba 函数，用于测试 dict 嵌套 dict
@njit
def create_dict_nested_dict():
    # 创建一个包含字典的字典
    outer_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=Dict.empty(key_type=types.unicode_type, value_type=types.int64),
    )

    # 创建并填充内部字典
    inner_dict_1 = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    inner_dict_1["key1"] = 1
    inner_dict_1["key2"] = 2

    inner_dict_2 = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    inner_dict_2["key3"] = 3
    inner_dict_2["key4"] = 4

    # 将内部字典添加到外部字典中
    outer_dict["inner_A"] = inner_dict_1
    outer_dict["inner_B"] = inner_dict_2

    return outer_dict


# 定义 Numba 函数
@njit
def create_nested_list_of_dicts():
    result_list = List.empty_list(
        Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    )

    # 创建并填充内部字典
    inner_dict_1 = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    inner_dict_1["key1"] = 1
    inner_dict_1["key2"] = 2

    inner_dict_2 = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    inner_dict_2["key3"] = 3
    inner_dict_2["key4"] = 4

    result_list.append(inner_dict_1)
    result_list.append(inner_dict_2)

    return result_list


# 定义一个新的 Numba 函数，用于测试 dict 嵌套 list
@njit
def create_dict_nested_list():
    # 创建一个包含列表的字典
    result_dict = Dict.empty(
        key_type=types.unicode_type, value_type=List.empty_list(types.int64)
    )

    # 创建并填充内部列表
    inner_list_1 = List.empty_list(types.int64)
    inner_list_1.append(1)
    inner_list_1.append(2)

    inner_list_2 = List.empty_list(types.int64)
    inner_list_2.append(3)
    inner_list_2.append(4)

    # 将内部列表添加到字典中
    result_dict["list_A"] = inner_list_1
    result_dict["list_B"] = inner_list_2

    return result_dict


# 定义一个新的 Numba 函数，用于测试 list 嵌套 list 嵌套 dict
@njit
def create_nested_list_of_lists_of_dicts():
    # 正确的嵌套语法：直接使用 empty() 方法嵌套
    outer_list = List.empty_list(
        List.empty_list(Dict.empty(key_type=types.unicode_type, value_type=types.int64))
    )

    # 第一个内层列表
    inner_list_1 = List.empty_list(
        Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    )
    inner_dict_1 = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    inner_dict_1["A"] = 1
    inner_dict_1["B"] = 2
    inner_list_1.append(inner_dict_1)

    # 第二个内层列表
    inner_list_2 = List.empty_list(
        Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    )
    inner_dict_2 = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    inner_dict_2["C"] = 3
    inner_dict_2["D"] = 4
    inner_list_2.append(inner_dict_2)

    # 将内层列表添加到外层列表
    outer_list.append(inner_list_1)
    outer_list.append(inner_list_2)

    return outer_list


# 定义一个新的 Numba 函数，用于测试 dict 嵌套 dict 嵌套 list
@njit
def create_dict_nested_dict_nested_list():
    # 创建一个包含字典嵌套列表的字典
    outer_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=Dict.empty(
            key_type=types.unicode_type, value_type=List.empty_list(types.int64)
        ),
    )

    # 创建并填充第一个中间字典
    middle_dict_1 = Dict.empty(
        key_type=types.unicode_type, value_type=List.empty_list(types.int64)
    )
    inner_list_1 = List.empty_list(types.int64)
    inner_list_1.append(100)
    inner_list_1.append(200)
    middle_dict_1["numbers_A"] = inner_list_1

    # 创建并填充第二个中间字典
    middle_dict_2 = Dict.empty(
        key_type=types.unicode_type, value_type=List.empty_list(types.int64)
    )
    inner_list_2 = List.empty_list(types.int64)
    inner_list_2.append(300)
    inner_list_2.append(400)
    middle_dict_2["numbers_B"] = inner_list_2

    # 将两个中间字典添加到外部字典
    outer_dict["data_1"] = middle_dict_1
    outer_dict["data_2"] = middle_dict_2

    return outer_dict


# 主测试块
if __name__ == "__main__":
    # 测试 list 嵌套 list
    run_and_verify(create_nested_list_of_lists, "list 嵌套 list")

    # 测试 dict 嵌套 dict
    run_and_verify(create_dict_nested_dict, "dict 嵌套 dict")

    # 测试 list 嵌套 dict
    run_and_verify(create_nested_list_of_dicts, "list 嵌套 dict")

    # 测试 dict 嵌套 list
    run_and_verify(create_dict_nested_list, "dict 嵌套 list")

    # 测试 list 嵌套 list 嵌套 dict
    run_and_verify(create_nested_list_of_lists_of_dicts, "list 嵌套 list 嵌套 dict")

    # 测试 dict 嵌套 dict 嵌套 list
    run_and_verify(create_dict_nested_dict_nested_list, "dict 嵌套 dict 嵌套 list")
