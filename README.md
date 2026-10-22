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

# 3. 启动服务
uv run uvicorn web.main:app --reload
```

## 目录结构

```
doc/         # 待解析文档（PDF/Markdown），可用 DATA_BASED_ROOT_DIR 覆盖
output/      # 解析产物（md/json/图片），可用 OUTPUT_DIR 覆盖
config/      # 配置模块（path_config.py 提供统一路径）
processor/   # 文档处理
utils/       # 通用工具
web/         # Web 服务入口
tool/        # 工具脚本
test/        # 测试
```

> 所有本地路径均从项目根目录推导（见 `config/path_config.py`），
> 不再硬编码 `D:\doc`、`D:/output` 等 Windows 路径，macOS / Linux / Windows 均可直接运行。
