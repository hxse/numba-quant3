from pathlib import Path
from src.utils.common import time_it, assert_attr_is_not_none


class DataArchiver:
    def _archive_data(self):
        """
        将数据存档。
        """
        assert_attr_is_not_none(self, "symbol", "data_path", "period_list", "data_list")

        server_dir = f"{self.symbol}/{self.period_list[0]}"
        local_dir = self.get_local_dir(self.data_path, server_dir)
        username, password = self.get_token(Path(f"{self.data_path}/config.json"))
        self.archive_data(
            self.data_list,
            save_local_dir=local_dir,
            save_zip_dir=local_dir,
            upload_server="http://127.0.0.1:5123",
            server_dir=server_dir,
            username=username,
            password=password,
        )
