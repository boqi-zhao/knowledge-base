# config/reranker_config.py

from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RerankerConfig:
    text_rerank_api_key: str = field(repr=False)
    text_rerank_model: str
    text_rerank_instruct: str
    text_rerank_api_base: str
    text_rerank_timeout: float

reranker_config = RerankerConfig(
    text_rerank_api_key=os.getenv("TEXT_RERANK_API_KEY", "").strip(),
    text_rerank_model=os.getenv("TEXT_RERANK_MODEL", "qwen3-rerank"),
    text_rerank_instruct=os.getenv("TEXT_RERANK_INSTRUCT", ""),
    text_rerank_api_base=os.getenv("TEXT_RERANK_API_BASE", "https://dashscope.aliyuncs.com/api/v1").rstrip("/"),
    text_rerank_timeout=float(os.getenv("TEXT_RERANK_TIMEOUT", "60")),
)
