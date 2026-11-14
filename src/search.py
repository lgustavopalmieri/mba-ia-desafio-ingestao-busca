import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
)
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "document_pdf")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-6-luna")

K_RESULTS = 10

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_vector_store() -> PGVector:
    return PGVector(
        embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


def retrieve_context(question: str, k: int = K_RESULTS) -> str:
    """Vetoriza a pergunta, busca os k chunks mais similares e os concatena."""
    results = get_vector_store().similarity_search_with_score(question, k=k)
    return "\n\n".join(document.page_content for document, _score in results)


def search_prompt(question: str) -> str:
    """Monta o prompt final com o contexto recuperado e a pergunta do usuário."""
    return PROMPT_TEMPLATE.format(contexto=retrieve_context(question), pergunta=question)


def answer_question(question: str) -> str:
    """Executa o fluxo completo: recupera contexto, monta o prompt e chama a LLM."""
    llm = ChatOpenAI(model=LLM_MODEL)
    response = llm.invoke(search_prompt(question))
    return response.content
