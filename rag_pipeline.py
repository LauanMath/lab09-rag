#!/usr/bin/env python3
"""
Lab 09 — Arquitetura RAG Avançada: HNSW + HyDE + Cross-Encoder
Assistente de busca semântica em manuais médicos.

Uso:
    export OPENAI_API_KEY="sk-..."
    python rag_pipeline.py
"""

import os
from openai import OpenAI

from corpus import MEDICAL_DOCS
from embeddings import embed
from hnsw_index import build_hnsw_index, hnsw_search
from hyde import generate_hypothetical_document
from reranker import cross_encoder_rerank


def run_pipeline(query: str, client: OpenAI) -> None:
    sep = "─" * 65

    # ── PASSO 1: Construção do Índice HNSW ───────────────────────────────────
    print(sep)
    print("PASSO 1 — Construção do Índice HNSW")
    print(sep)
    print(f"  Corpus  : {len(MEDICAL_DOCS)} fragmentos de manual médico")
    print("  Gerando embeddings …")

    doc_vecs = embed(MEDICAL_DOCS, client)
    print(f"  Shape dos embeddings : {doc_vecs.shape}")

    index = build_hnsw_index(doc_vecs)
    print(f"  Índice HNSW pronto   | ntotal={index.ntotal}")

    # ── PASSO 2: HyDE — Documento Hipotético ─────────────────────────────────
    print(f"\n{sep}")
    print("PASSO 2 — HyDE: Geração do Documento Hipotético")
    print(sep)
    print(f"  Query coloquial : {query}")
    print("  Gerando documento hipotético via LLM …")

    hyp_doc = generate_hypothetical_document(query, client)
    print(f"\n  Documento hipotético:\n    {hyp_doc}")

    hyde_vec = embed([hyp_doc], client)[0]

    # ── PASSO 3: Busca Rápida no HNSW — Top-10 ───────────────────────────────
    print(f"\n{sep}")
    print("PASSO 3 — Recuperação via HNSW (Bi-Encoder | Top-10 | Funil Largo)")
    print(sep)

    scores, idxs = hnsw_search(hyde_vec, index, k=10)
    top10: list[str] = []
    for rank, (score, idx) in enumerate(zip(scores, idxs), 1):
        doc = MEDICAL_DOCS[idx]
        top10.append(doc)
        preview = doc[:110] + ("…" if len(doc) > 110 else "")
        print(f"\n  [{rank:02d}] cosseno={score:.4f}")
        print(f"       {preview}")

    # ── PASSO 4: Re-ranking com Cross-Encoder — Top-3 ────────────────────────
    print(f"\n{sep}")
    print("PASSO 4 — Re-ranking com Cross-Encoder (Top-3 Finais)")
    print(sep)
    print("  Aplicando atenção cruzada nos 10 candidatos …\n")

    top3 = cross_encoder_rerank(query, top10, top_k=3)
    print("  Documentos finais — injetados no contexto do LLM gerador:\n")
    for rank, (score, doc) in enumerate(top3, 1):
        print(f"  [{rank}] cross-encoder score = {score:.4f}")
        print(f"      {doc}\n")


def main() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    query = "dor de cabeça latejante e luz incomodando muito"
    run_pipeline(query, client)


if __name__ == "__main__":
    main()
