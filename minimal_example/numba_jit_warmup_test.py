import time
from numba import njit, types
from numba.typed import List
from numba.core.types import unicode_type
import multiprocessing
import sys

# 假设 numba_config 已被正确定义
numba_config = {
    "enable_cache": False,
    "nb": {"float": types.float64, "int": types.int64},
}
enable_cache = numba_config["enable_cache"]


# JIT工具函数
@njit(cache=enable_cache)
def get_length_from_list(data_list):
    return len(data_list)


@njit(cache=enable_cache)
def get_item_from_list(data_list, num):
    return data_list[num]


# convert_keys 函数，保持原样
def convert_keys(keys):
    n = get_length_from_list(keys)
    split_keys = [get_item_from_list(keys, i).split("_")[0] for i in range(n)]
    return tuple(dict.fromkeys(split_keys))


# 测试数据生成函数，保持原样
params_list_type = types.ListType(unicode_type)


@njit(params_list_type(), cache=enable_cache)
def get_list_empty():
    _list = List.empty_list(types.unicode_type)
    return _list


@njit(params_list_type(), cache=enable_cache)
def get_list_one():
    _list = List.empty_list(types.unicode_type)
    _list.append("test_one")
    return _list


@njit(params_list_type(), cache=enable_cache)
def get_keys():
    _l = List.empty_list(types.unicode_type)
    for i in ("keys_1", "keys_2", "keys_3"):
        _l.append(i)
    return _l


# --- 将每个场景封装到单独的函数中 ---


def run_empty_warmup_scenario():
    print("--- 场景一：用空列表预热 ---")
    start_time_empty_warmup = time.perf_counter()
    convert_keys(get_list_empty())
    get_length_from_list(get_list_empty())
    end_time_empty_warmup = time.perf_counter()
    print(f"空列表预热耗时：{end_time_empty_warmup - start_time_empty_warmup:.6f} 秒")

    signal_keys = get_keys()
    start_time_after_empty = time.perf_counter()
    print(f"信号键的长度: {get_length_from_list(signal_keys)}")
    print(f"转换后的键: {convert_keys(signal_keys)}")
    end_time_after_empty = time.perf_counter()
    print(
        f"空列表预热后，处理实际数据耗时：{end_time_after_empty - start_time_after_empty:.6f} 秒\n"
    )


def run_one_warmup_scenario():
    print("--- 场景二：用非空列表预热 ---")
    start_time_one_warmup = time.perf_counter()
    convert_keys(get_list_one())
    get_length_from_list(get_list_one())
    end_time_one_warmup = time.perf_counter()
    print(f"非空列表预热耗时：{end_time_one_warmup - start_time_one_warmup:.6f} 秒")

    signal_keys = get_keys()
    start_time_after_one = time.perf_counter()
    print(f"信号键的长度: {get_length_from_list(signal_keys)}")
    print(f"转换后的键: {convert_keys(signal_keys)}")
    end_time_after_one = time.perf_counter()
    print(
        f"非空列表预热后，处理实际数据耗时：{end_time_after_one - start_time_after_one:.6f} 秒"
    )


if __name__ == "__main__":
    # 在Windows上，默认就是'spawn'，无需显式设置
    # multiprocessing.freeze_support() 是必须的
    multiprocessing.freeze_support()

    p1 = multiprocessing.Process(target=run_empty_warmup_scenario)
    p2 = multiprocessing.Process(target=run_one_warmup_scenario)

    p1.start()
    p1.join()

    p2.start()
    p2.join()
