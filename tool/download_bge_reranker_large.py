"""历史本地 BGE 重排模型下载工具；当前业务使用百炼 Reranker API，无需运行。"""

import os

from dotenv import load_dotenv
from modelscope.hub.snapshot_download import snapshot_download

from config.path_config import PROJECT_ROOT

load_dotenv()

# 模型缓存目录：优先读取 .env 中的 MODELSCOPE_CACHE，默认项目根目录 models/modelscope_cache
cache_root = os.getenv("MODELSCOPE_CACHE") or str(PROJECT_ROOT / "models" / "modelscope_cache")

# 下载 BGE Reranker Large 模型
model_dir = snapshot_download('BAAI/bge-reranker-large', cache_dir=os.path.join(cache_root, "models"))
print(f"模型已下载到：{model_dir}")
