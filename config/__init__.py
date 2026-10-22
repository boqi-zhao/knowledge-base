"""
config 包入口。

在任何 config.* 子模块被导入时，先执行一次路径引导：
把模型 / 缓存目录默认指向项目内（见 config/path_config.py），
避免 huggingface、modelscope、torch、tiktoken 等库把文件写到 $HOME 或 /tmp。
"""

from config import path_config  # noqa: F401  (导入即执行路径引导)
from config.path_config import (  # noqa: F401
    CACHE_DIR,
    DOC_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    TEMP_DIR,
    ensure_dir,
    project_path,
)
