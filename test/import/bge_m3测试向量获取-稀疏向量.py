from utils.embedding_utils import generate_embeddings

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
embeddings = generate_embeddings(docs)
sparse = embeddings["sparse"]

print(sparse)
#milvus数据库存储稀疏向量的格式要求
# {
#     key: "列"，
#     value: "值"
# }

