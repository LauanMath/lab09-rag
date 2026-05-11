import os

# Resposta pré-gerada pelo gpt-4o-mini para a query de demonstração.
# Mantida aqui para garantir reprodutibilidade quando a API não está disponível.
_HYDE_FALLBACK = (
    "Cefaleia pulsátil unilateral (enxaqueca) caracteriza-se por dor latejante de "
    "intensidade moderada a grave, tipicamente unilateral, com duração de 4 a 72 horas "
    "sem tratamento. O quadro frequentemente é acompanhado de fotofobia — hipersensibilidade "
    "patológica à luz que causa desconforto ocular intenso e piora da cefaleia em ambientes "
    "iluminados — e fonofobia. Náuseas e vômitos podem ocorrer nas crises mais intensas. "
    "O diagnóstico é clínico, conforme os critérios da ICHD-3 para enxaqueca sem aura."
)

LLM_MODEL = "gpt-4o-mini"

_PROMPT_TEMPLATE = (
    "Você é um especialista em semiologia médica. "
    "Dado o relato coloquial de um paciente, escreva o fragmento que um manual clínico "
    "usaria para descrever esse quadro, com terminologia médica precisa. "
    "Produza apenas o fragmento — não responda ao paciente.\n\n"
    'Relato do paciente: "{query}"\n\n'
    "Fragmento do manual médico:"
)


def generate_hypothetical_document(query: str) -> str:
    """
    HyDE — Hypothetical Document Embeddings.
    Tenta gerar o documento via OpenAI API; usa resposta pré-gerada como fallback
    para garantir reprodutibilidade do pipeline sem dependência de chave ativa.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": _PROMPT_TEMPLATE.format(query=query)}],
                temperature=0.3,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [HyDE] API indisponível ({e}). Usando documento pré-gerado.")

    print(f"  [HyDE] Usando documento hipotético pré-gerado por {LLM_MODEL}.")
    return _HYDE_FALLBACK
