from utils.embedding_utils import generate_embeddings

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
embeddings = generate_embeddings(docs)
dense_list = embeddings["dense"]

print(dense_list)
