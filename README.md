# knowledge-base

基于 FastAPI + LangChain / LangGraph 构建的知识库服务，支持文档处理、向量检索与 RAG 问答。

## 技术栈

- FastAPI + Uvicorn
- LangChain / LangGraph
- Milvus（向量库，可选）
- MinIO（对象存储）
- MongoDB（元数据）

## 快速开始

```bash
# 1. 安装依赖（使用 uv）
uv sync

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入实际配置

# 3. 启动服务（导入服务 8000 / 查询服务 8001）
uv run python web/api/import_service.py
uv run python web/api/query_service.py
```

## Embedding 配置

Embedding 使用百炼 `text-embedding-v4` 的原生 API，同时返回 1024 维稠密向量和稀疏向量，不加载本地 BGE 模型。
在 `.env` 中单独填写 `EMBEDDING_API_KEY`，与对话模型的 `OPENAI_API_KEY` 独立：

```dotenv
EMBEDDING_API_KEY=你的百炼Key
EMBEDDING_API_BASE=https://dashscope.aliyuncs.com/api/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIM=1024
EMBEDDING_TIMEOUT=60
```

接口须使用原生地址，不能填 `compatible-mode/v1`（不返回稀疏向量）。
`generate_embeddings(texts)` 保持原有 `dense` 列表和 `sparse` 字典列表格式，业务节点、Milvus 集合和混合检索逻辑不变。
如果已有 BGE-M3 生成的入库数据，需重新生成文档及商品名向量；同为 1024 维也不能混用不同模型的向量。
本次配置切换不会自动重建或删除数据库数据。

## Reranker 配置

重排使用百炼 `qwen3-rerank` 原生 API，不加载本地 BGE 重排模型，也不依赖 DashScope SDK。
在 `.env` 的 `Reranker API` 分组配置 `TEXT_RERANK_API_KEY`（可使用与 Embedding 相同的百炼 Key）、
`TEXT_RERANK_API_BASE=https://dashscope.aliyuncs.com/api/v1`、`TEXT_RERANK_MODEL=qwen3-rerank` 和 `TEXT_RERANK_TIMEOUT=60`。
`TEXT_RERANK_INSTRUCT` 配置排序任务指令，留空使用模型默认指令。
`rerank_documents(query, documents)` 继续按输入文档顺序返回分数；业务节点的排序与动态 Top-K 截取逻辑保持不变。
修改配置后重启服务。

## 目录结构

```
doc/         # 待解析文档（PDF/Markdown），可用 DATA_BASED_ROOT_DIR 覆盖
output/      # 解析产物（md/json/图片），可用 OUTPUT_DIR 覆盖
models/      # 本地模型 / HF・ModelScope 缓存，可用 MODELS_DIR 覆盖
.cache/      # torch / tiktoken / XDG 等缓存，可用 CACHE_DIR 覆盖
temp-files/  # 临时文件，可用 MD_ROOT_DIR 覆盖
config/      # 配置模块（path_config.py 提供统一路径）
processor/   # 文档处理
utils/       # 通用工具
web/         # Web 服务入口
tool/        # 工具脚本
test/        # 测试
```

## 路径约定（关键）

**所有本地文件一律写在项目目录内**，不会写入用户家目录、/tmp 或系统目录。

`config/path_config.py` 在导入时（`config/__init__.py` 触发）会做两件事：

1. 把 `doc / output / models / .cache / temp-files` 定位到项目根目录；
2. 通过 `os.environ.setdefault` 引导第三方库的缓存位置：

| 环境变量 | 默认位置 | 用途 |
|---|---|---|
| `HF_HOME` | `models/huggingface_cache` | HuggingFace 模型 |
| `MODELSCOPE_CACHE` | `models/modelscope_cache` | ModelScope 模型 |
| `SENTENCE_TRANSFORMERS_HOME` | `models/sentence-transformers` | sentence-transformers |
| `TORCH_HOME` | `.cache/torch` | PyTorch 预训练权重 |
| `TIKTOKEN_CACHE_DIR` | `.cache/tiktoken` | tiktoken BPE 文件 |
| `XDG_CACHE_HOME` | `.cache` | 其他遵循 XDG 的库 |

需要调整时在 `.env` 中覆盖同名变量即可；相对路径按项目根目录解析。

查看当前生效的路径：

```bash
uv run python tool/check_paths.py
```
