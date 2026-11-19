# processor/import_processor/nodes/node_item_name_recognition.py
import json
import logging
import uuid
from typing import Tuple, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pymilvus import DataType

from config.milvus_config import milvus_config
from config.path_config import OUTPUT_DIR
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.prompt.item_name_recognition import (
    ITEM_NAME_EVIDENCE_TEMPLATE,
    ITEM_NAME_RECOGNITION_SYSTEM_PROMPT,
    ITEM_NAME_RECOGNITION_TEMPLATE,
)
from processor.import_processor.state import ImportGraphState, ItemNameStatus
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.markdown_utils import strip_inline_images
from utils.milvus_utils import get_milvus_client, escape_milvus_string


ITEM_NAME_MAX_BYTES = 100


class NodeItemNameRecognition(BaseNode):
    """从各切片摘录证据，识别整篇文档的主体设备。"""

    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        file_title, chunks = self._step_1_get_inputs(state)
        item_name, status = self._recognize_item_name(file_title, chunks)
        self._step_4_update_chunks(state, chunks, item_name)
        state["item_name_status"] = status

        dense_vector, sparse_vector = self._step_5_generate_vectors(item_name)
        self._step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        self.logger.info("主体识别完成，状态=%s，名称=%s", status, item_name)
        return state

    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, List[Dict]]:
        """
        步骤 1: 接收并校验流程输入
        核心作用：
            1. 从流程状态中提取文件标题、文本切片核心数据
            2. 基础数据类型校验，保证下游流程输入有效性
        依赖的状态数据（上游节点产出）：
            - state["file_title"]: 上游提取的文件标题
            - state["chunks"]: 文本切片列表
        返回值：
            Tuple[str, List[Dict]]: (处理后的文件标题, 校验后的文本切片列表)
        """

        # 1、参数校验
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        return file_title, chunks

    def _recognize_item_name(
        self, file_title: str, chunks: list[dict]
    ) -> tuple[str, ItemNameStatus]:
        try:
            llm = get_llm_client(json_mode=True)
            evidence = self._collect_evidence(llm, chunks)
            self.logger.info("主体证据覆盖 %d/%d 个切片", len(evidence), len(chunks))
            if not evidence:
                return file_title, "unconfirmed"

            document = {
                "file_title": file_title,
                "evidence": [
                    {"chunk_index": index, "quote": quote} for index, quote in evidence.items()
                ],
            }
            prompt = ITEM_NAME_RECOGNITION_TEMPLATE.format(
                document=json.dumps(document, ensure_ascii=False)
            )
            result = self._invoke_json(llm, prompt)
            status = result.get("status")
            if status == "multiple":
                return file_title, "multiple"
            if status == "unconfirmed":
                return file_title, "unconfirmed"
            if status != "identified":
                raise ValueError("无效的主体识别状态")

            item_name = result.get("item_name")
            evidence_chunk = result.get("evidence_chunk")
            if type(evidence_chunk) is not int or evidence_chunk not in evidence:
                raise ValueError("主体名称缺少有效的证据切片编号")
            if not isinstance(item_name, str) or len(item_name.strip().splitlines()) != 1:
                raise ValueError("主体名称必须是非空的单行文本")
            item_name = "".join(item_name.split())
            if not 0 < len(item_name.encode("utf-8")) <= ITEM_NAME_MAX_BYTES:
                raise ValueError("主体名称超出存储长度")
            return item_name, "identified"

        except Exception as exc:
            self.logger.warning("主体识别失败，使用文件标题兜底（%s）", type(exc).__name__)
            return file_title, "unconfirmed"

    def _collect_evidence(self, llm: ChatOpenAI, chunks: list[dict]) -> dict[int, str]:
        evidence = {}
        for index, chunk in enumerate(chunks, start=1):
            # 图片替代文本可能来自上游模型生成的摘要。
            original_title = chunk.get("title") or ""
            original_content = chunk.get("content") or ""
            title = strip_inline_images(original_title).strip()
            content = strip_inline_images(original_content).strip()
            if not title and not content:
                continue

            prompt = ITEM_NAME_EVIDENCE_TEMPLATE.format(
                chunk=json.dumps({"title": title, "content": content}, ensure_ascii=False)
            )
            quote = self._invoke_json(llm, prompt)["quote"]
            if quote is None:
                continue
            if not isinstance(quote, str) or not quote.strip():
                raise ValueError("主体证据必须是非空原文或 null")

            quote = quote.strip()
            if not (
                (quote in title and quote in original_title)
                or (quote in content and quote in original_content)
            ):
                self.logger.warning("丢弃第 %d 个切片中找不到的主体证据", index)
                continue
            evidence[index] = quote
        return evidence

    @staticmethod
    def _invoke_json(llm: ChatOpenAI, prompt: str) -> dict:
        response = llm.invoke([
            SystemMessage(content=ITEM_NAME_RECOGNITION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        result = json.loads(response.content)
        if not isinstance(result, dict):
            raise ValueError("主体识别模型必须返回 JSON 对象")
        return result

    def _step_4_update_chunks(self, state: ImportGraphState, chunks: List[Dict[str, str]], item_name: str):
        """
        步骤 4: 回填商品名称到流程状态和所有文本切片
        核心作用：
            1. 全局状态更新：将item_name存入state，供下游所有节点直接使用
            2. 切片数据补全：为每个切片添加item_name字段，保证数据一致性
            3. 状态同步：更新state中的chunks，确保切片修改全局生效
        设计思路：
            所有切片关联同一商品名称，保证后续向量入库、检索时的维度一致性
        参数：
            state: 流程状态对象
            chunks: 校验后的文本切片列表
            item_name: 识别出的商品名称，或兜底的文件标题
        """

        state["item_name"] = item_name

        for chunk in chunks:
            chunk["item_name"] = item_name

        state["chunks"] = chunks

    def _step_5_generate_vectors(self, item_name: str) -> Tuple[List, Dict]:
        """
        步骤 5: 通过百炼原生 Embedding API 为商品名称生成稠密 + 稀疏双向量
        核心说明：
            - 模型由 EMBEDDING_MODEL 配置，当前使用 text-embedding-v4
            - 稠密向量（dense_vector）：浮点数列表，当前业务集合要求 EMBEDDING_DIM=1024
            - 稀疏向量（sparse_vector）：{词表维度索引: 权重} 字典，已在工具层做 L2 归一化
        依赖工具：
            generate_embeddings：接收文本列表，调用百炼 API，校验结果并按输入顺序返回双向量
        参数：
            item_name: 步骤 3 识别的商品名称；为空时直接返回 (None, None)
        返回值：
            (稠密向量列表, 稀疏向量字典)，名称为空时返回 (None, None)
        """

        # 1. 名称为空时返回 (None, None)，不调用 Embedding API
        if not item_name:
            return None, None

        # 2. 将单个名称包装为文本列表，调用 API 后取回对应的第一组双向量
        vectors = generate_embeddings([item_name])
        return vectors["dense"][0], vectors["sparse"][0]

    def _step_6_save_to_milvus(self, state: ImportGraphState, file_title: str, item_name: str, dense_vector,
                               sparse_vector):
        """
        步骤 6: 将商品名称、文件标题、双向量持久化到 Milvus 向量数据库
        核心逻辑：
            1. 客户端获取：获取单例 Milvus 客户端，连接失败则跳过
            2. 集合初始化：无集合则创建（定义 Schema+索引），有集合则直接使用
            3. 幂等性处理：删除同名商品数据，避免重复存储
            4. 数据插入：构造符合 Schema 的数据，非空向量才添加
        参数：
            state: 流程状态对象，用于最终状态同步
            file_title: 处理后的文件标题
            item_name: 识别后的商品名称（主键去重依据）
            dense_vector: 步骤 5 生成的稠密向量（1024 维列表）
            sparse_vector: 步骤 5 生成的稀疏向量（字典格式）
        """


        # 1. 获取 Milvus 客户端
        milvus_client = get_milvus_client()
        if not milvus_client:
            # 降级处理
            self.logger.warning("Milvus 连接失败，跳过设备名称保存")
            return
        try:
            # 2. 集合初始化
            # kb_item_names
            collection_name = milvus_config.item_name_collection

            # 判断集合是否已经被创建
            if not milvus_client.has_collection(collection_name):
                self._create_item_name_collection(collection_name, milvus_client)

            # 3. 幂等性处理：删除同名商品数据，避免重复存储
            file_title = escape_milvus_string(file_title) # Milvus数据库过滤表达式中字符串的安全转义函数（防止解析失败）
            milvus_client.delete(collection_name=collection_name, filter=f"file_title=='{file_title}'")

            # 4. 数据插入：构造符合 Schema 的数据，非空向量才添加
            data = {
                "file_title": file_title,
                "item_name": item_name
            }

            if dense_vector is not None:
                data["dense_vector"] = dense_vector
            if sparse_vector is not None:
                data["sparse_vector"] = sparse_vector

            milvus_client.insert(collection_name=collection_name, data=data)


        except Exception as e:
            # 降级处理
            self.logger.warning(f"数据存入Milvus失败，跳过设备名称保存，错误原因:{str(e)}")

    def _create_item_name_collection(self, collection_name, milvus_client):

        # 1. 创建schema
        schema = milvus_client.create_schema(
            auto_id=True, 
            enable_dynamic_field=True
        )

        # 2. 创建字段
        # 添加主键字段（INT64类型，自增）
        schema.add_field(
            field_name="pk",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )

        # 添加文件标题字段（VARCHAR类型，最大长度100字节）
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=100
        )

        # 添加商品名称字段（VARCHAR类型，最大长度100字节）
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=ITEM_NAME_MAX_BYTES
        )

        # 添加稠密向量字段（FLOAT_VECTOR类型，1024维，与当前 Embedding 模型设置的输出维度一致）
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024
        )

        # 添加稀疏向量字段（SPARSE_FLOAT_VECTOR类型，接收工具层转换后的索引与权重字典）
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # 3. 构建索引参数（提升向量检索性能）
        index_params = milvus_client.prepare_index_params()
        # 为稠密向量创建索引（IVF_FLAT：兼容性好，适合小数据量）
        # 核心是 “先聚类分桶、再桶内暴力精确检索”。
        index_params.add_index(
            field_name="dense_vector",  # 字段名
            index_name="dense_vector_index",  # 索引名
            index_type="IVF_FLAT",  # 索引类型（兼容所有Milvus版本）
            metric_type="COSINE",  # 相似度计算方式（余弦相似度）---关注方向
            params={"nlist": 128}  # 聚类数（影响检索精度/速度）
        )

        # 为稀疏向量创建索引（SPARSE_INVERTED_INDEX：稀疏向量专用索引）
        index_params.add_index(
            field_name="sparse_vector",  # 字段名
            index_name="sparse_vector_index",  # 索引名
            index_type="SPARSE_INVERTED_INDEX",  # 索引类型
            metric_type="IP",  # 相似度计算方式（内积）---即关注方向，也关注大小
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                # 高效的稀疏检索算法

                "normalize": True,
                # 稀疏向量已在 generate_embeddings 中做 L2 归一化；非零单位向量的 IP 等于余弦相似度

                "quantization": "none"
                # 量化参数设为 none；输入权重来自百炼 API，已转换为 Python float
            })

        # 4. 创建集合（Schema + 索引）
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )


if __name__ == "__main__":

    setup_logging()

    demo_title = "H3C LA2608室内无线网关 用户手册-6W100-整本手册"
    json_path = OUTPUT_DIR / demo_title / "chunks.json"
    with open(json_path, "r", encoding="utf-8") as f:
        chuncks_json = f.read()

    chuncks = json.loads(chuncks_json)

    init_state = {
        "task_id": str(uuid.uuid4()),
        "chunks": chuncks,
        "file_title": demo_title
    }

    # 执行核心处理流程
    node_item_name_recognition = NodeItemNameRecognition()
    result = node_item_name_recognition(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))
