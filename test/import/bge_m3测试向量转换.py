from utils.embedding_utils import generate_embeddings

docs = [
    "hello.",
    "再见",
    "苹果",
]

docs_embeddings = generate_embeddings(docs)

print(docs_embeddings)
