import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
)
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "document_pdf")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def ingest_pdf() -> None:
    if not os.path.exists(PDF_PATH):
        print(f"Arquivo não encontrado: {PDF_PATH}")
        return

    documents = PyPDFLoader(PDF_PATH).load()
    print(f"{len(documents)} páginas carregadas de {PDF_PATH}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"{len(chunks)} chunks gerados ({CHUNK_SIZE} caracteres, overlap {CHUNK_OVERLAP})")

    vector_store = PGVector(
        # task_type ajusta o vetor para o uso pretendido pelo Gemini Embedding
        embeddings=GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, task_type="retrieval_document"
        ),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
        # Recria a collection a cada ingestão: a tabela de vetores é criada na
        # primeira execução com a dimensão do modelo de embeddings escolhido;
        # sem isso, trocar de modelo causa erro de incompatibilidade de dimensão.
        pre_delete_collection=True,
    )
    vector_store.add_documents(chunks)
    print(f"Ingestão concluída: {len(chunks)} chunks salvos na collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as error:
        print(f"Falha na ingestão: {error}")
        print(
            "Verifique se o banco está ativo (docker compose up -d) "
            "e se GOOGLE_API_KEY está configurada no .env"
        )
