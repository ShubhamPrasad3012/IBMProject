from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np

def build_vector_index(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(chunks)

    dim = vectors[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))

    return index, chunks, model

def get_top_chunks(query, model, index, chunks, k=3):
    q_vec = model.encode([query])
    _, I = index.search(np.array(q_vec), k)
    return "\n".join([chunks[i] for i in I[0]])
