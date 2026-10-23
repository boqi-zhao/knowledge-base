import os
from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    api_key: str = field(repr=False)
    api_base: str
    model: str
    dimension: int
    timeout: float

embedding_config = EmbeddingConfig(
    api_key=os.getenv("EMBEDDING_API_KEY", "").strip(),
    api_base=os.getenv("EMBEDDING_API_BASE", "https://dashscope.aliyuncs.com/api/v1").rstrip("/"),
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
    dimension=int(os.getenv("EMBEDDING_DIM", "1024")),
    timeout=float(os.getenv("EMBEDDING_TIMEOUT", "60")),
)
