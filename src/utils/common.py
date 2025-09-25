import time
from contextlib import contextmanager


def assert_attr_is_not_none(self, *attrs: str):
    """
    检查多个实例属性是否为 None。
    """
    for attr in attrs:
        if getattr(self, attr) is None:
            raise ValueError(f"缺少必要的属性: '{attr}'")


@contextmanager
def time_it(show_timing, message):
    """
    一个用于计时的上下文管理器。
    """
    if show_timing:
        start_time = time.perf_counter()
        # print(f"{message} 开始...")
    try:
        yield
    finally:
        if show_timing:
            end_time = time.perf_counter()
            duration = end_time - start_time
            print(f"{message} 运行时间: {duration:.4f} 秒")
