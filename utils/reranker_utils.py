"""重排工具统一转发到百炼 API，不再初始化本地 BGE 模型。"""

from utils.reranker_http_utils import rerank_documents

__all__ = ["rerank_documents"]
