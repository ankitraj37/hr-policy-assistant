# HR Policy Assistant (MVP)

An AI-powered chatbot that answers employee questions **only** from company HR policy PDFs using Retrieval-Augmented Generation (RAG).

**Employees never upload documents** – the administrator places PDFs in the `data/` folder and runs the ingestion script.

## Features

- Strict RAG: answers are grounded only in retrieved policy chunks
- Citations with document name + page number
- Confidence score (High / Medium / Low)
- Conversation history within the session
- Analytics dashboard (total questions, avg confidence, most referenced policy)
- Clean Streamlit UI with no upload controls for employees

## Tech Stack

| Component          | Technology                          |
|--------------------|-------------------------------------|
| Frontend           | Streamlit                           |
| LLM                | Google Gemini (gemini-2.0-flash)    |
| Vector DB          | ChromaDB (persistent)               |
| Embeddings         | sentence-transformers / all-MiniLM-L6-v2 |
| Document loading   | PyPDF + LangChain                   |
| Config             | python-dotenv                       |

## Architecture
