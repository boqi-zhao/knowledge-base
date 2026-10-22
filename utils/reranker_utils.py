import os
# from FlagEmbedding import FlagReranker

from config.path_config import resolve_path

_reranker_model = None
def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        # 模型路径支持相对写法（相对项目根目录），统一解析为绝对路径
        model_path = resolve_path(os.getenv("BGE_RERANKER_LARGE"))
        _reranker_model= FlagReranker(
            model_name_or_path=str(model_path) if model_path else None,
            device=os.getenv("BGE_RERANKER_DEVICE" ),
            use_fp16=os.getenv("BGE_RERANKER_FP16" )
        )
    return _reranker_model