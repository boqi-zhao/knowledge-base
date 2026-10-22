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
config/      # 配置模块
processor/   # 文档处理
utils/       # 通用工具
web/         # Web 服务入口
tool/        # 工具脚本
test/        # 测试
```
