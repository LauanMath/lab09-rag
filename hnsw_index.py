import numpy as np
import faiss

# M: número de ligações bidirecionais por nó — controla recall e uso de RAM do grafo
# ef_construction: lista dinâmica durante o build — afeta qualidade sem custo em produção
HNSW_M          = 32
EF_CONSTRUCTION = 200


def build_hnsw_index(embeddings: np.ndarray) -> faiss.IndexHNSWFlat:
    """
    Constrói índice FAISS-HNSW com similaridade de cosseno.
    Para vetores normalizados (L2=1), produto interno == cosseno.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.add(embeddings)
    return index


def hnsw_search(
    query_vec: np.ndarray,
    index: faiss.IndexHNSWFlat,
    k: int = 10,
    ef_search: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna os k vizinhos mais próximos (maior cosseno) no grafo HNSW."""
    index.hnsw.efSearch = ef_search
    q = query_vec.reshape(1, -1).copy()
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    return scores[0], idxs[0]
