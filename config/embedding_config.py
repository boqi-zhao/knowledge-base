import os
from dataclasses import dataclass

from config.path_config import resolve_path

# 模型路径支持相对写法（相对项目根目录），统一解析为绝对路径
_bge_m3_path = resolve_path(os.getenv("BGE_M3_PATH"))


@dataclass
class EmbeddingConfig:
    bge_m3_path: str
    bge_m3: str
    bge_device: str
    bge_fp16: bool

embedding_config = EmbeddingConfig(
    bge_m3_path=str(_bge_m3_path) if _bge_m3_path else "",
    bge_m3=os.getenv("BGE_M3"),
    bge_device=os.getenv("BGE_DEVICE"),
    # 特殊处理：将.env中的1/0转为布尔值，兼容常见的数字/字符串格式
    bge_fp16=os.getenv("BGE_FP16") in ("1", "True", "true", 1)
)
