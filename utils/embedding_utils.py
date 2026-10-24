import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config.embedding_config import embedding_config


class EmbeddingAPIError(RuntimeError):
    """Embedding 调用失败或服务未返回有效的双向量。"""


def generate_embeddings(texts):
    """通过百炼原生 API 生成双向量，稀疏权重统一做 L2 归一化后返回。"""
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
        # text_index 是当前批次内的输入文本序号，用于恢复 API 返回条目的顺序；
        # 它与下面 sparse_embedding 中表示词表维度的 index 含义不同。
        index = entry.get("text_index")
        if type(index) is not int or not 0 <= index < batch_size or dense_vectors[index] is not None:
            raise EmbeddingAPIError("百炼 Embedding API 返回了重复或无效的 text_index")
        # 本地 BGE-M3 的 dense 是 NumPy 数组，需要 .tolist()；
        # 百炼的 embedding 已经是 JSON 数值列表，仍校验为现有业务所需的 1024 维。
        dense = entry.get("embedding")
        if (not isinstance(dense, list) or len(dense) != embedding_config.dimension
                or not all(_is_finite_number(value) for value in dense)):
            raise EmbeddingAPIError("百炼 Embedding API 返回了错误维度或无效的稠密向量")
        # 稀疏格式差异：本地 BGE-M3 返回整个批次的 CSR 稀疏矩阵，原来需要：
        #   start, end = sparse_obj.indptr[i], sparse_obj.indptr[i + 1]
        #   dict(zip(sparse_obj.indices[start:end], sparse_obj.data[start:end]))
        # 百炼不返回 data / indices / indptr，而是为每条文本单独返回：
        #   sparse_embedding = [{"index": 102897, "value": 1.1865, "token": "无线"}, ...]
        # 结构对应：indices 的行切片 -> 每项 index；data 的行切片 -> 每项 value。
        # indptr 原本负责划分文本的行边界；API 已按文本分组，无需再切片，
        # 再用外层 text_index 将这一组向量对应回输入文本即可。
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
            # token 是便于阅读的词项文本；业务只使用 {整数维度索引: 权重}。
            sparse_dict[token_index] = float(value)
        # 两种调用方式转换后的数据结构相同，但模型的词表索引含义、权重及
        # 稠密向量空间不同，百炼向量不能与旧 BGE-M3 入库向量混用。
        dense_vectors[index] = [float(value) for value in dense]
        # 入库和查询共用此入口：都将稀疏向量归一化为单位长度，保持字典格式。
        sparse_vectors[index] = normalize_sparse_vector(sparse_dict)
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

    # L2 范数 = sqrt(sum(value ** 2))；hypot 避免极小/极大权重平方时下溢/上溢。
    l2_norm = math.hypot(*sparse_vec.values())
    if l2_norm == 0:  # 全零向量无法归一化，保留原值，避免除以 0。
        return sparse_vec

    # 每个权重除以同一个范数，列索引与权重比例不变；输出仍为 Python float。
    return {index: float(value / l2_norm) for index, value in sparse_vec.items()}
