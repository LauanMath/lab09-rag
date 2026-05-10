import numpy as np
import faiss
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"


def embed(texts: list[str], client: OpenAI) -> np.ndarray:
    """
    Gera vetores normalizados (L2=1) para uma lista de textos via OpenAI Embeddings.
    Vetores unitários permitem usar produto interno como similaridade de cosseno.
    """
    resp = client.embeddings.create(input=texts, model=EMBED_MODEL)
    vecs = np.array([e.embedding for e in resp.data], dtype=np.float32)
    faiss.normalize_L2(vecs)
    return vecs
