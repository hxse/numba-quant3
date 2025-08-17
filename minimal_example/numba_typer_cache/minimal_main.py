import sys
from pathlib import Path

parentPath = Path(__file__).parent
if parentPath:
    sys.path.insert(0, str(parentPath))


import typer
import re

from minimal_config import config


def main(cache: bool = True):
    # 更新全局 config 字典
    config["numba_cache"] = cache
    print(f"cache from cli: {cache}")

    # 在解析参数并更新 config 后再导入 numba_functions
    from minimal_numba_functions import your_numba_function

    # 调用 Numba 函数
    result = your_numba_function(10)
    print(f"Numba 函数结果: {result}")


class AliasGroup(typer.core.TyperGroup):
    _CMD_SPLIT_P = re.compile(r" ?[,|] ?")

    def get_command(self, ctx, cmd_name):
        cmd_name = self._group_cmd_name(cmd_name)
        return super().get_command(ctx, cmd_name)

    def _group_cmd_name(self, default_name):
        for cmd in self.commands.values():
            name = cmd.name
            if name and default_name in self._CMD_SPLIT_P.split(name):
                return name
        return default_name


if __name__ == "__main__":
    app = typer.Typer(
        cls=AliasGroup, pretty_exceptions_show_locals=False, no_args_is_help=True
    )
    app.command("main | ma")(main)
    app()
