"""
检查当前生效的路径配置，确认所有本地文件都落在项目目录内。

用法：
    uv run python tool/check_paths.py
"""

import sys
from pathlib import Path

# 允许直接运行本文件：把项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from config.path_config import PROJECT_ROOT, describe  # noqa: E402

if __name__ == "__main__":
    info = describe()
    for key, value in info.items():
        flag = "OK " if value.startswith(str(PROJECT_ROOT)) else "!! "
        print(f"{flag}{key:<28}{value}")

    outside = [
        (k, v) for k, v in info.items()
        if k not in ("PROJECT_ROOT",) and not v.startswith(str(PROJECT_ROOT))
    ]
    print()
    if outside:
        print(f"[警告] 以下 {len(outside)} 项指向项目目录之外：")
        for k, v in outside:
            print(f"  {k} = {v}")
    else:
        print("[通过] 所有路径均位于项目目录内 ✓")
