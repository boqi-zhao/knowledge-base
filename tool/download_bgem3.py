import os

from dotenv import load_dotenv
from modelscope.hub.snapshot_download import snapshot_download

from config.path_config import PROJECT_ROOT

load_dotenv()

# 模型缓存目录：优先读取 .env 中的 MODELSCOPE_CACHE，默认项目根目录 models/modelscope_cache
cache_root = os.getenv("MODELSCOPE_CACHE") or str(PROJECT_ROOT / "models" / "modelscope_cache")

model_dir = snapshot_download('BAAI/bge-m3', cache_dir=os.path.join(cache_root, "models"))
print(f"模型已下载到：{model_dir}")
