# Desafio MBA Engenharia de Software com IA — Full Cycle

## Ingestão e Busca Semântica com LangChain e Postgres

Sistema de **RAG (Retrieval-Augmented Generation)** que lê um PDF, armazena seu conteúdo em um banco **PostgreSQL + pgVector** e responde perguntas via **CLI**, usando apenas o conteúdo do documento como fonte de verdade.

### Como funciona

1. **Ingestão** (`src/ingest.py`): o PDF é carregado com `PyPDFLoader`, dividido em chunks de **1000 caracteres com overlap de 150** (`RecursiveCharacterTextSplitter`) e cada chunk é convertido em embedding (`text-embedding-3-small`). Os vetores são salvos no PostgreSQL com a extensão **pgVector** (`langchain-postgres`).
2. **Busca** (`src/search.py`): a pergunta é vetorizada e os **10 chunks mais similares** são recuperados com `similarity_search_with_score(k=10)`.
3. **Resposta** (`src/chat.py`): os chunks recuperados são concatenados em um prompt com regras rígidas (responder apenas com base no contexto; nunca inventar) e enviados à LLM (`gpt-6-luna`), que devolve a resposta ao usuário.

```
┌─────────────┐   ┌──────────────┐   ┌──────────────────┐
│ document.pdf│ → │   ingest.py  │ → │ PostgreSQL+pgVector│
└─────────────┘   │ (split+embed)│   │  (vetores)        │
                  └──────────────┘   └────────┬─────────┘
                                              │ k=10 mais similares
┌─────────────┐   ┌──────────────┐            │
│  chat.py    │ → │  search.py   │ ←──────────┘
│ (CLI)       │   │ (prompt+LLM) │
└─────────────┘   └──────────────┘
```

### Tecnologias

| Camada            | Ferramenta                                                         |
| ----------------- | ------------------------------------------------------------------ |
| Linguagem         | Python 3                                                           |
| Framework         | LangChain (`langchain`, `langchain-community`, etc.)               |
| Banco de dados    | PostgreSQL + pgVector (via `langchain-postgres`)                   |
| Execução do banco | Docker & Docker Compose                                            |
| PDF               | `PyPDFLoader` (`langchain_community.document_loaders`)             |
| Split             | `RecursiveCharacterTextSplitter` (`langchain_text_splitters`)      |
| Embeddings        | `OpenAIEmbeddings` (`langchain_openai`) — `text-embedding-3-small` |
| LLM               | `ChatOpenAI` (`langchain_openai`) — `gpt-6-luna`                   |

Os modelos foram escolhidos consultando a [documentação oficial da OpenAI](https://developers.openai.com/api/docs/models): `text-embedding-3-small` é o modelo de embeddings atual e mais barato, e `gpt-6-luna` é o modelo de texto mais eficiente/barato disponível. Ambos podem ser trocados editando as variáveis `OPENAI_EMBEDDING_MODEL` e `OPENAI_LLM_MODEL` no `.env`.

### Pré-requisitos

- [Docker](https://docs.docker.com/get-docker/) e Docker Compose
- Python **3.11, 3.12 ou 3.13** (recomendado: **3.13**)
- Uma [API key da OpenAI](https://platform.openai.com/api-keys)

### Preparando o Python 3.13

Use um dos canais abaixo para obter o 3.13:

#### Ubuntu (PPA deadsnakes)

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.13 python3.13-venv
python3.13 --version
```

#### Qualquer sistema (uv)

Baixa um Python 3.13 oficial automaticamente, sem sudo — funciona igual em Linux, macOS e Windows:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

cd mba-ia-desafio-ingestao-busca   # ajuste para o caminho onde está o repositório
uv venv --python 3.13 venv
source venv/bin/activate    # no Windows: venv\Scripts\activate
python --version            # deve imprimir 3.13.x
```

#### Windows/macOS sem uv

- **Windows**: instale o Python 3.13 pelo [site oficial](https://www.python.org/downloads/) (marque "Add python.exe to PATH"), crie o venv com `py -3.13 -m venv venv` e ative com `venv\Scripts\activate`.
- **macOS**: `brew install python@3.13` e crie o venv com `python3.13 -m venv venv`.

### Como executar

1. **Crie e ative o ambiente virtual com o Python 3.13**

   ```bash
   python3.13 -m venv venv
   source venv/bin/activate
   python --version   # deve imprimir 3.13.x — se for 3.14, veja "Preparando o Python 3.13"
   ```

   No Windows, use `py -3.13 -m venv venv` e ative com `venv\Scripts\activate`.

2. **Instale as dependências**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure as variáveis de ambiente**

   ```bash
   cp .env.example .env
   ```

   Edite o `.env` e preencha a sua chave:

   ```
   OPENAI_API_KEY=sk-...
   ```

   Os demais valores já vêm com padrões compatíveis com o `docker-compose.yml`.

4. **Suba o banco de dados**

   ```bash
   docker compose up -d
   ```

   Este comando inicia o PostgreSQL com a extensão pgVector habilitada. Para verificar:

   ```bash
   docker compose ps
   ```

5. **Execute a ingestão do PDF**

   ```bash
   python src/ingest.py
   ```

   Saída esperada (valores aproximados):

   ```
   34 páginas carregadas de document.pdf
   67 chunks gerados (1000 caracteres, overlap 150)
   Ingestão concluída: 67 chunks salvos na collection 'document_pdf'
   ```

6. **Rode o chat**

   ```bash
   python src/chat.py
   ```

   Exemplo de uso:

   ```
   Chat iniciado. Digite sua pergunta ou 'sair' para encerrar.

   PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
   RESPOSTA: O faturamento foi de R$ 10.000.000,00.

   PERGUNTA: Quantos clientes temos em 2024?
   RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

   PERGUNTA: sair
   Encerrando o chat. Até mais!
   ```

### Estrutura do projeto

```
.
├── docker-compose.yml   # PostgreSQL + pgVector (com healthcheck e extensão vector)
├── requirements.txt     # Dependências com versões fixadas
├── .env.example         # Template das variáveis de ambiente
├── .env                 # Suas variáveis (não versionado — está no .gitignore)
├── document.pdf         # PDF a ser ingerido
├── README.md
└── src/
    ├── ingest.py        # Carrega o PDF, gera chunks e embeddings, salva no pgVector
    ├── search.py        # Recupera os 10 chunks mais relevantes, monta o prompt e chama a LLM
    └── chat.py          # CLI de interação com o usuário
```

### Solução de problemas

| Sintoma                                                    | Causa provável                                                    | Solução                                                                                                             |
| ---------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `Não foi possível responder...`                            | Banco desligado ou `OPENAI_API_KEY` vazia                         | `docker compose up -d` e confira o `.env`                                                                           |
| Erro de dimensão do vetor                                  | Trocou o modelo de embeddings após a ingestão                     | Execute `python src/ingest.py` novamente (recria a collection)                                                      |
| Banco em estado inconsistente                              | Dados antigos de testes                                           | `docker compose down -v && docker compose up -d` e refaça a ingestão                                                |
| `ModuleNotFoundError: No module named 'dotenv'`            | `pip install` não concluiu (venv vazio — normalmente Python 3.14) | Confira `python --version` (3.13), recrie o venv e rode `pip install -r requirements.txt`                           |
| `Unable to locate package python3.13`                      | PPA deadsnakes não registrado                                     | Rode `sudo add-apt-repository ppa:deadsnakes/ppa -y && sudo apt update` e confira com `apt-cache policy python3.13` |
| `No matching distribution found for psycopg-binary==3.2.9` | venv criado com Python 3.14                                       | Recrie o venv com `python3.13 -m venv venv`                                                                         |

### Comandos úteis

```bash
# Ver os logs do banco
docker compose logs postgres

# Inspecionar os vetores salvos
docker exec -it postgres_rag psql -U postgres -d rag -c \
  "SELECT collection_id, COUNT(*) FROM langchain_pg_embedding GROUP BY collection_id;"

# Parar tudo (mantém os dados no volume)
docker compose down

# Parar tudo e apagar o volume (recomeçar do zero)
docker compose down -v
```
