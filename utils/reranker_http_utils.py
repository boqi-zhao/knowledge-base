import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config.reranker_config import reranker_config


def rerank_documents(query: str, documents: list[str]) -> list[float]:
    """调用百炼重排 API，按 documents 原始顺序返回分数，供业务节点自行排序。"""
    if not isinstance(documents, list):
        raise TypeError("documents 必须是文本列表")
    if not documents:
        return []
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Reranker 查询必须是非空字符串")
    if any(not isinstance(doc, str) or not doc.strip() for doc in documents):
        raise ValueError("Reranker 文档必须是非空字符串")
    if len(documents) > 500:
        raise ValueError("百炼 qwen3-rerank 单次最多支持 500 条文档")
    if not reranker_config.text_rerank_api_key:
        raise ValueError("请在 .env 中配置百炼 TEXT_RERANK_API_KEY")

    parameters = {"top_n": len(documents), "return_documents": False}
    if reranker_config.text_rerank_instruct:
        parameters["instruct"] = reranker_config.text_rerank_instruct
    payload = {
        "model": reranker_config.text_rerank_model,
        "input": {"query": query, "documents": documents},
        "parameters": parameters,
    }
    request = Request(
        reranker_config.text_rerank_api_base + "/services/rerank/text-rerank/text-rerank",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + reranker_config.text_rerank_api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=reranker_config.text_rerank_timeout) as response:
            body = json.load(response)
    except HTTPError as error:
        status = error.code
        error.close()
        raise RuntimeError(f"百炼 Reranker API 请求失败，HTTP {status}") from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("百炼 Reranker API 连接失败或超时") from None
    except (ValueError, UnicodeError):
        raise RuntimeError("百炼 Reranker API 返回了无效 JSON") from None

    output = body.get("output") if isinstance(body, dict) else None
    results = output.get("results") if isinstance(output, dict) else None
    if not isinstance(results, list) or len(results) != len(documents):
        raise RuntimeError("百炼 Reranker API 返回的分数数量与文档数不一致")
    scores = [None] * len(documents)
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError("百炼 Reranker API 返回了无效的重排结果")
        index = result.get("index")
        score = result.get("relevance_score")
        if type(index) is not int or not 0 <= index < len(documents) or scores[index] is not None:
            raise RuntimeError("百炼 Reranker API 返回了重复或无效的文档索引")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError("百炼 Reranker API 返回了无效的相关性分数")
        scores[index] = float(score)
    return scores
