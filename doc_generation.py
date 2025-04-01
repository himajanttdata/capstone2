import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class DocGeneration:
    def __init__(self, index_dir="faiss_indexes"):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

    def store_vectors(self, repo_name, chunks, model_name="all-MiniLM-L6-v2"):
        vectors = self.embed_chunks(chunks, model_name)
        index_path = os.path.join(self.index_dir, f"{repo_name}.index")

        dimension = vectors.shape[1]
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.add(vectors)
        faiss.write_index(index, index_path)

    def embed_chunks(self, chunks, model_name):
        model = SentenceTransformer(model_name)
        vectors = model.encode(chunks)
        return np.array(vectors, dtype=np.float32)
