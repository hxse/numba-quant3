import httpx
import json
import io
import time
from pathlib import Path

_TOKEN_CACHE = {}


def get_local_dir(data_path, server_dir):
    return f"{data_path}/output/{server_dir}"


def get_token(token_path):
    with open(token_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data["username"], data["password"]


def request_token(
    client: httpx.Client, upload_server: str, username: str | None, password: str | None
) -> str | None:
    """
    请求服务器获取访问令牌。

    参数:
    client (httpx.Client): 一个已初始化的 httpx 客户端实例。
    upload_server (str): 服务器的 URL。
    username (str): 用于认证的用户名。
    password (str): 用于认证的密码。

    返回:
    str | None: 获取到的 access_token，如果获取失败则返回 None。
    """
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "password",
        "client_id": "_",
        "client_secret": "",
        "username": username,
        "password": password,
    }

    try:
        response = client.post(
            f"{upload_server}/auth/token", data=data, headers=headers
        )
        response.raise_for_status()
        token_data = response.json()
        access_token = token_data.get("access_token")
        if access_token:
            _TOKEN_CACHE[(username, password)] = access_token  # 存储 token
            return access_token
        else:
            print("响应中没有找到 Access Token。")
            print("完整的响应内容：")
            print(json.dumps(token_data, indent=2))
            return None
    except httpx.HTTPError as err:
        print(f"HTTP 错误：{err}")
        print(f"响应内容：{err.response.text}")
        return None
    except httpx.RequestException as err:
        print(f"发生请求错误：{err}")
        return None


def upload_data(
    client: httpx.Client,
    upload_server: str,
    files: dict,
    username: str | None,
    password: str | None,
    max_retries: int = 3,
    wait: int = 1,
) -> None:
    """
    将文件上传到服务器，并处理 401 错误重试逻辑。

    参数:
    client (httpx.Client): 一个已初始化的 httpx 客户端实例。
    upload_server (str): 服务器的 URL。
    files (dict): 包含要上传文件的字典。
    username (str | None): 用户名，用于重新获取 token。
    password (str | None): 密码，用于重新获取 token。
    max_retries (int): 最大重试次数，默认为 3。

    返回:
    None: 函数不返回任何值。
    """
    retries = max_retries
    while retries >= 0:
        # 从缓存获取 token 或请求新 token 的逻辑
        access_token = _TOKEN_CACHE.get((username, password))
        if not access_token:
            access_token = request_token(client, upload_server, username, password)
            if not access_token:
                print("无法获取 Access Token，上传中止。")
                return

        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            response = client.post(
                f"{upload_server}/file/upload", files=files, headers=headers
            )
            response.raise_for_status()
            print("文件已成功上传到服务器。")
            return  # 上传成功，退出循环和函数
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                print(
                    "上传文件到服务器失败: HTTP 状态码 401 (Unauthorized)。尝试重新获取 Access Token 并重试。"
                )
                if (username, password) in _TOKEN_CACHE:
                    del _TOKEN_CACHE[(username, password)]  # 清空缓存中的 token
            else:
                print(f"上传文件到服务器失败: HTTP 状态码错误 - {e}")
        except (httpx.HTTPError, httpx.RequestError) as e:
            print(f"上传文件到服务器失败: 请求或HTTP错误 - {e}")
        except Exception as e:
            print(f"上传文件到服务器失败: 未知错误 - {e}")
            return  # 遇到未知错误，直接退出

        if retries > 0:
            print(f"剩余重试次数: {retries}")
            retries -= 1
            time.sleep(wait)  # 等待一秒后重试
        else:
            print("重试次数已用尽，上传中止。")
            return  # 重试次数用尽，退出函数


def upload_to_server(
    client: httpx.Client,
    upload_server: str,
    zip_data: bytes,
    server_dir: Path | None = None,
    zip_name: str = "strategy.zip",
    username: str | None = None,
    password: str | None = None,
):
    """
    将 ZIP 文件的字节数据上传到指定的服务器。

    参数:
    client (httpx.Client): 一个已初始化的 httpx 客户端实例。
    upload_server (str): 服务器的 URL。
    zip_data (bytes): ZIP 文件的字节数据。
    zip_dir_path (Path | None): ZIP 文件所在的目录路径，用于生成默认文件名。
    zip_name (str): ZIP 文件的名称，默认为 "strategy.zip"。
    username (str | None): 用于 HTTPBearer 认证的用户名。
    password (str | None): 用于 HTTPBearer 认证的密码。
    """
    start_time = time.perf_counter()

    file_path = (
        Path(server_dir) if server_dir else Path("./")
    ) / f"{zip_name if zip_name else 'temp.zip'}"

    zip_file_to_upload = io.BytesIO(zip_data)
    files = {
        "file": (
            file_path.as_posix(),
            zip_file_to_upload,
            "application/zip",
        )
    }

    # upload_data 不再返回布尔值，因此需要修改调用方式
    upload_data(client, upload_server, files, username, password)
    # 假设如果函数执行到这里没有抛出异常，则表示上传尝试完成
    # 具体的成功/失败信息会在 upload_data 内部打印
    time_elapsed = time.perf_counter() - start_time
    print(f"ZIP 文件上传过程完成，用时 {time_elapsed:.2f} 秒")
