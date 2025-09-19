import numpy as np

def convert_keys(keys):
    # 使用列表推导式获取所有元素
    split_keys = [i.split("_")[0] for i in keys]

    # 使用 dict.fromkeys() 高效去重并保留顺序
    unique_list = list(dict.fromkeys(split_keys))

    # 将结果转换为 tuple 并返回
    return tuple(unique_list)
