from numba import njit
from minimal_config import config

cache = config["numba_cache"]
print("cache for numba:", cache)


# 使用 config 字典中的缓存设置
@njit(cache=cache)
def your_numba_function(x):
    return x * 2
