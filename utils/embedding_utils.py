import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from config.embedding_config import embedding_config


class EmbeddingAPIError(RuntimeError):
    """Embedding 调用失败或服务未返回有效的双向量。"""


def generate_embeddings(texts):
    """通过百炼原生 API 生成双向量，保持原有 dense 列表 / sparse 字典列表合同。"""
    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts 必须是文本列表")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding 输入必须是非空字符串")
    if not texts:
        return {"dense": [], "sparse": []}
    if not embedding_config.api_key:
        raise ValueError("请在 .env 中配置百炼 EMBEDDING_API_KEY")
    if embedding_config.dimension != 1024:
        raise ValueError("现有业务集合使用 1024 维向量，请设置 EMBEDDING_DIM=1024")

    result = {"dense": [], "sparse": []}
    # text-embedding-v4 每次最多接收 10 条文本；内部拆批不改变调用方的输入顺序。
    for start in range(0, len(texts), 10):
        batch = texts[start:start + 10]
        payload = {
            "model": embedding_config.model,
            "input": {"texts": list(batch)},
            "parameters": {
                "dimension": embedding_config.dimension,
                "output_type": "dense&sparse",
                "text_type": "document",
            },
        }
        request = Request(
            embedding_config.api_base + "/services/embeddings/text-embedding/text-embedding",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + embedding_config.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=embedding_config.timeout) as response:
                body = json.load(response)
        except HTTPError as error:
            # 不输出请求头、Key 或服务端原始响应，避免凭证进入业务日志。
            status = error.code
            error.close()
            raise EmbeddingAPIError(f"百炼 Embedding API 请求失败，HTTP {status}") from None
        except (URLError, TimeoutError, OSError):
            raise EmbeddingAPIError("百炼 Embedding API 连接失败或超时") from None
        except (ValueError, UnicodeError):
            raise EmbeddingAPIError("百炼 Embedding API 返回了无效 JSON") from None

        dense, sparse = _parse_embeddings(body, len(batch))
        result["dense"].extend(dense)
        result["sparse"].extend(sparse)
    return result


def _parse_embeddings(body, batch_size):
    """按 text_index 恢复顺序，并在入库前校验维度及稀疏向量的整数索引。"""
    output = body.get("output") if isinstance(body, dict) else None
    entries = output.get("embeddings") if isinstance(output, dict) else None
    if not isinstance(entries, list) or len(entries) != batch_size:
        raise EmbeddingAPIError("百炼 Embedding API 返回的向量数量与输入不一致")

    dense_vectors = [None] * batch_size
    sparse_vectors = [None] * batch_size
    for entry in entries:
        if not isinstance(entry, dict):
            raise EmbeddingAPIError("百炼 Embedding API 返回了无效的向量条目")
        index = entry.get("text_index")
        if type(index) is not int or not 0 <= index < batch_size or dense_vectors[index] is not None:
            raise EmbeddingAPIError("百炼 Embedding API 返回了重复或无效的 text_index")
        dense = entry.get("embedding")
        if (not isinstance(dense, list) or len(dense) != embedding_config.dimension
                or not all(_is_finite_number(value) for value in dense)):
            raise EmbeddingAPIError("百炼 Embedding API 返回了错误维度或无效的稠密向量")
        sparse = entry.get("sparse_embedding")
        if not isinstance(sparse, list) or not sparse:
            raise EmbeddingAPIError("百炼未返回有效稀疏向量，请使用原生 API 及 dense&sparse 输出")
        sparse_dict = {}
        for item in sparse:
            if not isinstance(item, dict):
                raise EmbeddingAPIError("百炼 Embedding API 返回了无效的稀疏向量")
            token_index, value = item.get("index"), item.get("value")
            if (type(token_index) is not int or token_index < 0
                    or token_index in sparse_dict or not _is_finite_number(value)):
                raise EmbeddingAPIError("百炼 Embedding API 返回了无效的稀疏索引或权重")
            sparse_dict[token_index] = float(value)
        dense_vectors[index] = [float(value) for value in dense]
        sparse_vectors[index] = sparse_dict
    return dense_vectors, sparse_vectors


def _is_finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)



def normalize_sparse_vector(sparse_vec):
    """
    对稀疏向量做 L2 归一化（仅处理非零维度，不影响零维度）
    :param sparse_vec: 原始稀疏向量（dict 格式：{维度: 数值}）
    :return: 归一化后的稀疏向量
    """
    if not sparse_vec:  # 空向量直接返回
        return sparse_vec

    # 提取非零维度的数值
    values = np.array(list(sparse_vec.values()), dtype=np.float64)
    # 计算 L2 范数（避免除以 0）
    l2_norm = np.linalg.norm(values)
    if l2_norm < 1e-9:  # 范数接近 0 时，直接返回原向量（避免除零错误）
        return sparse_vec

    # 归一化：每个数值除以 L2 范数
    normalized_values = values / l2_norm
    # normalized_values = (values / l2_norm).astype(np.float32)  # 统一转为 float32
    # 重建稀疏向量 dict
    return dict(zip(sparse_vec.keys(), normalized_values))
