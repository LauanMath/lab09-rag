from sentence_transformers import CrossEncoder

CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def cross_encoder_rerank(
    query: str,
    candidates: list[str],
    top_k: int = 3,
) -> list[tuple[float, str]]:
    """
    Re-ranqueia candidatos com Cross-Encoder de atenção cruzada bidirecional.
    Formato de entrada ao modelo: [CLS] query [SEP] documento.
    Retorna os top_k pares (score, documento) ordenados por relevância.
    """
    ce = CrossEncoder(CROSS_ENCODER_NAME)
    pairs = [[query, doc] for doc in candidates]
    scores = ce.predict(pairs)
    ranked = sorted(zip(scores.tolist(), candidates), reverse=True)
    return ranked[:top_k]
