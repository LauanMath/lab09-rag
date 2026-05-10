from openai import OpenAI

LLM_MODEL = "gpt-4o-mini"

_PROMPT_TEMPLATE = (
    "Você é um especialista em semiologia médica. "
    "Dado o relato coloquial de um paciente, escreva o fragmento que um manual clínico "
    "usaria para descrever esse quadro, com terminologia médica precisa. "
    "Produza apenas o fragmento — não responda ao paciente.\n\n"
    'Relato do paciente: "{query}"\n\n'
    "Fragmento do manual médico:"
)


def generate_hypothetical_document(query: str, client: OpenAI) -> str:
    """
    HyDE — Hypothetical Document Embeddings.
    Pede ao LLM que alucine um trecho técnico de manual para a query coloquial,
    criando uma âncora geométrica no espaço vetorial do jargão médico.
    """
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": _PROMPT_TEMPLATE.format(query=query)}],
        temperature=0.3,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()
