import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"   # BERT-based model (HuggingFace)
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  [Embedding] Carregando modelo '{EMBED_MODEL}' …")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """
    Gera vetores normalizados (L2=1) usando modelo BERT (sentence-transformers).
    normalize_embeddings=True → produto interno == similaridade de cosseno.
    """
    model = _get_model()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)
