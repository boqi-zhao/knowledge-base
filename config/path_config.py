# config/path_config.py
"""
项目统一路径配置（所有本地文件一律落在项目目录内）

设计原则
--------
1. 所有「输入 / 输出 / 缓存 / 模型」目录都从项目根目录推导，
   不写入 $HOME、/tmp、系统目录，也不硬编码 D:\\doc 这类绝对路径。
2. 环境变量（.env）优先；.env 里写相对路径时，按项目根目录解析，
   因此无论从哪个工作目录启动进程，结果都一致。
3. 导入本模块即完成第三方库缓存目录的引导（huggingface / modelscope /
   torch / tiktoken / sentence-transformers / XDG），保证它们默认也不写到用户目录。

可通过 .env 覆盖的变量
----------------------
    DATA_BASED_ROOT_DIR  上传文档根目录      默认 <项目根>/doc
    OUTPUT_DIR           解析产物根目录      默认 <项目根>/output
    MD_ROOT_DIR          临时文件目录        默认 <项目根>/temp-files
    MODELS_DIR           模型目录根          默认 <项目根>/models
    CACHE_DIR            缓存目录根          默认 <项目根>/.cache
    MODELSCOPE_CACHE     ModelScope 模型缓存 默认 <CACHE_DIR>/modelscope/models
    HF_HOME              HuggingFace 缓存    默认 <CACHE_DIR>/huggingface
    TORCH_HOME           PyTorch 缓存        默认 <CACHE_DIR>/torch
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 项目根目录（本文件位于 <项目根>/config/ 下）
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def resolve_path(value, default: Path = None) -> Path:
    """把配置值解析为绝对路径：空值返回 default，相对路径按项目根目录解析。"""
    if not value:
        return default
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _resolve_path(env_key: str, default: Path) -> Path:
    """读取环境变量指定的路径；相对路径按项目根目录解析。"""
    return resolve_path(os.getenv(env_key), default)


# ---------------- 业务目录 ----------------
# 上传 / 待解析文档的根目录
DOC_DIR: Path = _resolve_path("DATA_BASED_ROOT_DIR", PROJECT_ROOT / "doc")
# 解析产物（markdown / json / 图片）的根目录
OUTPUT_DIR: Path = _resolve_path("OUTPUT_DIR", PROJECT_ROOT / "output")
# 临时文件目录
TEMP_DIR: Path = _resolve_path("MD_ROOT_DIR", PROJECT_ROOT / "temp-files")
# 模型根目录 / 缓存根目录
MODELS_DIR: Path = _resolve_path("MODELS_DIR", PROJECT_ROOT / "models")
CACHE_DIR: Path = _resolve_path("CACHE_DIR", PROJECT_ROOT / ".cache")


# ---------------- 第三方库缓存引导 ----------------
# 用 setdefault：.env 中显式配置的值优先，未配置时统一指向项目内。
_CACHE_ENV_DEFAULTS = {
    # HuggingFace
    "HF_HOME": MODELS_DIR / "huggingface_cache",
    "HF_HUB_CACHE": MODELS_DIR / "huggingface_cache" / "hub",
    "TRANSFORMERS_CACHE": MODELS_DIR / "huggingface_cache" / "hub",
    # ModelScope
    "MODELSCOPE_CACHE": MODELS_DIR / "modelscope_cache",
    # PyTorch
    "TORCH_HOME": CACHE_DIR / "torch",
    # tiktoken（langchain 调用 OpenAI 时会下载 BPE 文件，默认落在系统临时目录）
    "TIKTOKEN_CACHE_DIR": CACHE_DIR / "tiktoken",
    # sentence-transformers
    "SENTENCE_TRANSFORMERS_HOME": MODELS_DIR / "sentence-transformers",
    # 其余遵循 XDG 规范的第三方库
    "XDG_CACHE_HOME": CACHE_DIR,
}

for _key, _default in _CACHE_ENV_DEFAULTS.items():
    _value = os.getenv(_key)
    if _value:
        # .env 里写的相对路径归一化为「相对项目根」的绝对路径
        _p = Path(_value).expanduser()
        os.environ[_key] = str(_p if _p.is_absolute() else (PROJECT_ROOT / _p))
    else:
        os.environ[_key] = str(_default)


# ---------------- 工具函数 ----------------
def ensure_dir(path) -> Path:
    """创建目录（含父级）并返回绝对路径。"""
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_path(*parts: str) -> Path:
    """按项目根目录拼接路径，例如 project_path("doc", "a.pdf")。"""
    return PROJECT_ROOT.joinpath(*parts)


def describe() -> dict:
    """打印/返回当前生效的路径配置，便于排查。"""
    keys = [
        "PROJECT_ROOT", "DOC_DIR", "OUTPUT_DIR", "TEMP_DIR", "MODELS_DIR", "CACHE_DIR",
        "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "MODELSCOPE_CACHE",
        "TORCH_HOME", "TIKTOKEN_CACHE_DIR", "SENTENCE_TRANSFORMERS_HOME", "XDG_CACHE_HOME",
    ]
    return {k: str(globals().get(k) or os.getenv(k, "")) for k in keys}


if __name__ == "__main__":
    for k, v in describe().items():
        print(f"{k:<26} {v}")
