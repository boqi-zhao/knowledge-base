# config/path_config.py
"""
项目统一路径配置

所有「文档输入目录 / 处理产物输出目录」都从项目根目录推导，
不再在代码里硬编码 D:\\doc、D:/output 这类 Windows 绝对路径，便于跨平台运行。

环境变量（可选，优先级高于默认值）：
    DATA_BASED_ROOT_DIR  上传文档的根目录，默认 <项目根>/doc
    OUTPUT_DIR           解析产物（md/json/图片）根目录，默认 <项目根>/output
    MD_ROOT_DIR          临时文件目录，默认 <项目根>/temp-files
相对路径会按项目根目录解析。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 项目根目录（本文件位于 <项目根>/config/ 下）
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _resolve_dir(env_key: str, default: Path) -> Path:
    """读取环境变量指定的目录；未配置时使用项目内默认目录。"""
    raw = os.getenv(env_key)
    if not raw:
        return default
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# 上传 / 待解析文档的根目录，默认 <项目根>/doc
DOC_DIR: Path = _resolve_dir("DATA_BASED_ROOT_DIR", PROJECT_ROOT / "doc")

# 解析产物（markdown / json / 图片）的根目录，默认 <项目根>/output
OUTPUT_DIR: Path = _resolve_dir("OUTPUT_DIR", PROJECT_ROOT / "output")

# 临时文件目录，默认 <项目根>/temp-files
TEMP_DIR: Path = _resolve_dir("MD_ROOT_DIR", PROJECT_ROOT / "temp-files")


def ensure_dir(path) -> Path:
    """创建目录（含父级）并返回绝对路径。"""
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_path(*parts: str) -> Path:
    """按项目根目录拼接路径，例如 project_path("doc", "a.pdf")。"""
    return PROJECT_ROOT.joinpath(*parts)
