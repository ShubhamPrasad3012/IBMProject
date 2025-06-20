from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np

def build_vector_index(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("No text chunks generated. Input text may be empty or invalid.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(chunks)

    if len(vectors) == 0:
        raise ValueError("No vectors were generated. Embedding may have failed.")

    dim = vectors[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))

    return index, chunks, model

def get_top_chunks(query, model, index, chunks, k=5):
    q_vec = model.encode([query])
    _, I = index.search(np.array(q_vec), k)

    # Protect against out-of-range indices
    return "\n".join([chunks[i] for i in I[0] if i < len(chunks)])
