# Lab 09 — Arquitetura RAG Avançada (HNSW, HyDE e Cross-Encoders)

> Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Lauan Matheus.

---

## Objetivo

Implementar um pipeline de **Retrieval-Augmented Generation (RAG)** de nível de produção para busca semântica em manuais médicos, combinando três técnicas avançadas:

- **HNSW** — índice vetorial hierárquico para busca aproximada ultrarrápida
- **HyDE** — transformação de query coloquial em documento hipotético técnico via LLM
- **Cross-Encoder** — re-ranking de alta precisão por atenção cruzada

---

## Estrutura do Repositório

```
lab09-rag/
├── corpus.py        # Dataset: 22 fragmentos de manuais médicos
├── embeddings.py    # Módulo de geração de vetores (BERT/HuggingFace)
├── hnsw_index.py    # Passo 1 — construção e busca no índice HNSW
├── hyde.py          # Passo 2 — geração do documento hipotético (HyDE)
├── reranker.py      # Passo 4 — re-ranking com Cross-Encoder
├── rag_pipeline.py  # Pipeline completo (orquestra os 4 passos)
├── requirements.txt
└── README.md
```

---

## Como Executar

```bash
pip install -r requirements.txt

python rag_pipeline.py
```

> **OpenAI API (opcional):** Se `OPENAI_API_KEY` estiver configurado, o Passo 2 usa o `gpt-4o-mini` para gerar o documento hipotético em tempo real. Sem a chave, o pipeline usa o documento pré-gerado pelo mesmo modelo.

---

## Descrição do Pipeline

### Passo 1 — Construção do Índice HNSW

22 fragmentos de manuais médicos são convertidos em vetores densos com o modelo BERT `all-MiniLM-L6-v2` (HuggingFace / sentence-transformers) e indexados num grafo HNSW via FAISS. Os vetores são normalizados para unidade antes da indexação, de modo que o produto interno equivale à **similaridade de cosseno**.

```python
index = faiss.IndexHNSWFlat(dim, M=32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200
```

### Passo 2 — HyDE (Hypothetical Document Embeddings)

Quando o paciente digita `"dor de cabeça latejante e luz incomodando muito"`, o LLM (`gpt-4o-mini` via OpenAI API) é instruído a **alucinar** um trecho técnico de manual que descreveria esse quadro — por exemplo:

> *"Cefaleia pulsátil unilateral acompanhada de fotofobia intensa e fonofobia, compatível com episódio de enxaqueca sem aura…"*

Esse documento falso é vetorizado e serve como **âncora geométrica** no espaço dos embeddings técnicos, eliminando o *vocabulary mismatch* entre linguagem coloquial e terminologia clínica.

### Passo 3 — Busca Rápida no HNSW (Bi-Encoder, Top-10)

O vetor HyDE é comparado com todos os vetores indexados via busca aproximada de vizinhos mais próximos. Retorna os **Top-10** fragmentos mais similares (funil largo) — impresso no console.

### Passo 4 — Re-ranking com Cross-Encoder (Top-3)

Os 10 candidatos são re-ranqueados pelo modelo `cross-encoder/ms-marco-MiniLM-L-6-v2`, que processa o par `[CLS] query_original [SEP] documento` com atenção cruzada bidirecional. Os **Top-3** resultantes são os documentos injetados no contexto do LLM gerador.

---

## Análise: HNSW vs KNN — Impacto dos Hiperparâmetros na RAM

### Custo de memória do KNN exato

Na busca exata K-Nearest Neighbors, é necessário armazenar todos os vetores e calcular a distância de cada query contra o corpus inteiro:

```
RAM_KNN = N × dim × 4 bytes
```

Para N = 1 000 000 vetores com dim = 1536 (text-embedding-3-small):
```
RAM_KNN ≈ 1M × 1536 × 4 ≈ 5,9 GB
```

Tempo de busca: **O(N × dim)** por query — inviável em produção com grandes corpora.

### Custo de memória do HNSW

O HNSW adiciona uma estrutura de grafo em camadas sobre os vetores brutos. O custo extra depende de dois hiperparâmetros:

**M (número de ligações bidirecionais por nó)**

- Cada nó armazena até `2 × M` vizinhos (IDs inteiros de 4 bytes).
- Custo extra do grafo: `N × 2M × 4 bytes`
- M maior → mais conexões → melhor recall → **mais RAM no índice final**
- Faixa típica: 16–64

**ef_construction (tamanho da lista dinâmica na construção)**

- Controla quantos candidatos são avaliados ao inserir cada nó durante o *build*.
- Afeta apenas a **RAM temporária durante a indexação** — não o índice em produção.
- ef_construction maior → índice de maior qualidade (recall) → build mais lento.

### Comparação: N = 1M, dim = 1536, M = 32

| Componente         | Fórmula              | Tamanho     |
|--------------------|----------------------|-------------|
| Vetores brutos     | N × dim × 4          | ~5,9 GB     |
| Grafo HNSW (M=32)  | N × 2M × 4           | ~256 MB     |
| **Total HNSW**     |                      | **~6,2 GB** |
| **KNN exato**      | apenas vetores brutos | **~5,9 GB** |

O HNSW consome **ligeiramente mais RAM** que KNN puro (pela estrutura do grafo), mas reduz o tempo de busca de **O(N)** para **O(log N)**, tornando a consulta viável em produção. A economia real é de **tempo e CPU** — não de memória bruta.

**Resumo prático:**

| Hiperparâmetro   | Efeito na RAM          | Efeito na qualidade | Efeito em produção |
|------------------|------------------------|---------------------|-------------------|
| M maior          | Aumenta (grafo maior)  | Recall maior        | Busca mais lenta  |
| ef_construction maior | Temporário (build) | Índice melhor    | Sem impacto       |

Para grandes corpora (N > 100K), HNSW é a única opção viável; KNN exato se torna computacionalmente proibitivo.

---

## Dependências

```
openai>=1.30.0
faiss-cpu>=1.7.4
sentence-transformers>=2.7.0
numpy>=1.24.0
```
