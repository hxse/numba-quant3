# numba-quant3
  * `uv sync`
  * `uv run --offline python .\src\main.py`
  * `uv run --offline pytest`
# 限制
  * [使用 @jit 编译 Python 代码 — Numba 0+untagged.663.ge6a4027.dirty 文档 --- Compiling Python code with @jit — Numba 0+untagged.663.ge6a4027.dirty documentation](https://numba.readthedocs.io/en/stable/user/jit.html)
  * numba不能自动检测其他模块深层缓存的变化,算了,不用了
