import time
from numba import njit, types
from numba.typed import Dict, List

# 设置测试的元素数量
NUM_ELEMENTS = 500

# ---
# 测试1: 在JIT函数中创建和填充List
# ---


@njit
def create_and_fill_numba_list():
    """在nopython模式下创建并填充一个大型的Numba List。"""
    numba_list = List.empty_list(types.int64)
    for i in range(NUM_ELEMENTS):
        numba_list.append(i)
    return numba_list


# ---
# 测试2: 在JIT函数中创建和填充Dict
# ---


@njit
def create_and_fill_numba_dict():
    """在nopython模式下创建并填充一个大型的Numba Dict。"""
    numba_dict = Dict.empty(key_type=types.int64, value_type=types.int64)
    for i in range(NUM_ELEMENTS):
        numba_dict[i] = i
    return numba_dict


# ---
# 测试3: 在JIT函数中创建和填充List嵌套Dict
# ---


@njit
def create_and_fill_nested_list_of_dicts():
    """在nopython模式下创建并填充List(Dict)。"""
    result_list = List.empty_list(
        Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    )
    for i in range(NUM_ELEMENTS):
        inner_dict = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
        inner_dict["value"] = i  # 使用一个固定字符串键
        result_list.append(inner_dict)
    return result_list


# ---
# 测试4: 在JIT函数中创建和填充Dict嵌套List
# ---


@njit
def create_and_fill_nested_dict_of_lists():
    """在nopython模式下创建并填充Dict(List)。"""
    result_dict = Dict.empty(
        key_type=types.int64, value_type=List.empty_list(types.int64)
    )
    for i in range(NUM_ELEMENTS):
        inner_list = List.empty_list(types.int64)
        inner_list.append(i)
        result_dict[i] = inner_list
    return result_dict


if __name__ == "__main__":
    print(f"--- Numba 容器性能测试 (元素数量: {NUM_ELEMENTS:,}) ---")
    print("\n")

    # ==========================
    # 列表 (List) 性能测试
    # ==========================
    print(">> 列表 (List) 性能")
    start_time = time.perf_counter()
    numba_list_result = create_and_fill_numba_list()
    jit_list_time = time.perf_counter() - start_time
    print(f"   JIT函数内部创建和填充时间: {jit_list_time:.6f} 秒")

    start_time = time.perf_counter()
    # py_list = list(numba_list_result)
    py_list = tuple(numba_list_result)
    conversion_list_time = time.perf_counter() - start_time
    print(f"   Numba List到Python List的转换时间: {conversion_list_time:.6f} 秒")

    start_time = time.perf_counter()
    total_sum_access = 0
    for i in range(NUM_ELEMENTS):
        total_sum_access += numba_list_result[i]
    access_list_time = time.perf_counter() - start_time
    print(f"   在Python层面上访问Numba List的每个元素: {access_list_time:.6f} 秒")
    print("\n")

    # ==========================
    # 字典 (Dict) 性能测试
    # ==========================
    print(">> 字典 (Dict) 性能")
    start_time = time.perf_counter()
    numba_dict_result = create_and_fill_numba_dict()
    jit_dict_time = time.perf_counter() - start_time
    print(f"   JIT函数内部创建和填充时间: {jit_dict_time:.6f} 秒")

    start_time = time.perf_counter()
    py_dict = dict(numba_dict_result)
    conversion_dict_time = time.perf_counter() - start_time
    print(f"   Numba Dict到Python Dict的转换时间: {conversion_dict_time:.6f} 秒")

    start_time = time.perf_counter()
    total_sum_access = 0
    for i in range(NUM_ELEMENTS):
        total_sum_access += numba_dict_result[i]
    access_dict_time = time.perf_counter() - start_time
    print(f"   在Python层面上访问Numba Dict的每个元素: {access_dict_time:.6f} 秒")
    print("\n")

    # ==========================
    # List(Dict) 性能测试
    # ==========================
    print(">> List(Dict) 性能")
    start_time = time.perf_counter()
    numba_list_of_dicts_result = create_and_fill_nested_list_of_dicts()
    jit_list_of_dicts_time = time.perf_counter() - start_time
    print(f"   JIT函数内部创建和填充时间: {jit_list_of_dicts_time:.6f} 秒")

    start_time = time.perf_counter()
    # py_list_of_dicts = list(numba_list_of_dicts_result)
    py_list_of_dicts = [dict(i) for i in tuple(numba_list_of_dicts_result)]
    conversion_list_of_dicts_time = time.perf_counter() - start_time
    print(
        f"   Numba List(Dict)到Python List(Dict)的转换时间: {conversion_list_of_dicts_time:.6f} 秒"
    )

    start_time = time.perf_counter()
    total_sum_access = 0
    for i in range(NUM_ELEMENTS):
        total_sum_access += numba_list_of_dicts_result[i]["value"]
    access_list_of_dicts_time = time.perf_counter() - start_time
    print(
        f"   在Python层面上访问Numba List(Dict)的每个元素: {access_list_of_dicts_time:.6f} 秒"
    )
    print("\n")

    # ==========================
    # Dict(List) 性能测试
    # ==========================
    print(">> Dict(List) 性能")
    start_time = time.perf_counter()
    numba_dict_of_lists_result = create_and_fill_nested_dict_of_lists()
    jit_dict_of_lists_time = time.perf_counter() - start_time
    print(f"   JIT函数内部创建和填充时间: {jit_dict_of_lists_time:.6f} 秒")

    start_time = time.perf_counter()
    py_dict_of_lists = dict(numba_dict_of_lists_result)
    conversion_dict_of_lists_time = time.perf_counter() - start_time
    print(
        f"   Numba Dict(List)到Python Dict(List)的转换时间: {conversion_dict_of_lists_time:.6f} 秒"
    )

    start_time = time.perf_counter()
    total_sum_access = 0
    for i in range(NUM_ELEMENTS):
        total_sum_access += numba_dict_of_lists_result[i][0]
    access_dict_of_lists_time = time.perf_counter() - start_time
    print(
        f"   在Python层面上访问Numba Dict(List)的每个元素: {access_dict_of_lists_time:.6f} 秒"
    )
    print("\n")
