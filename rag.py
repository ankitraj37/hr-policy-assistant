"""
rag.py
------
Core Retrieval-Augmented Generation logic.
- Loads the persistent ChromaDB
- Retrieves top-k relevant chunks
- Calls Gemini with a strict system prompt
- Returns answer + sources + confidence
"""

import os
from typing import List, Dict, Tuple
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "hr_policies"

# Strict system prompt
SYSTEM_PROMPT = """You are an HR Policy Assistant.
Answer ONLY using the provided policy content.
If the answer is not present in the retrieved documents, respond:
'I could not find this information in the available HR policies.'
Never invent policies.
Never assume company rules.
Always cite sources.
Always include document name and page number when possible.
Keep answers concise, clear, and professional.
"""


class HRPolicyRAG:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self._initialized = False

    def initialize(self) -> Tuple[bool, str]:
        """
        Initialize embeddings, vector store and Gemini LLM.
        Returns (success: bool, message: str)
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key.strip() in ("", "YOUR_GEMINI_API_KEY", "PASTE_YOUR_GEMINI_API_KEY_HERE"):
            return False, "Gemini API key not found. Please add GEMINI_API_KEY to your .env file."

        if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
            return False, (
                "Knowledge base is empty. "
                "Please place PDF files in the data/ folder and run: python ingest.py"
            )

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            self.vectorstore = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME,
            )

            count = self.vectorstore._collection.count()
            if count == 0:
                return False, "ChromaDB collection is empty. Please re-run ingest.py."

            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3.6-flash",
                google_api_key=api_key,
                temperature=0.1,
                max_output_tokens=1024,
            )

            self._initialized = True
            return True, f"Ready. Knowledge base contains {count} chunks."

        except Exception as e:
            return False, f"Initialization failed: {str(e)}"

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k most relevant chunks with similarity scores."""
        if not self._initialized:
            return []

        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)

        retrieved = []
        for doc, score in results:
            retrieved.append({
                "content": doc.page_content,
                "file_name": doc.metadata.get("file_name", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "score": float(score),
            })
        return retrieved

    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """Call Gemini with the retrieved context."""
        if not retrieved_chunks:
            return "I could not find this information in the available HR policies."

        # Build context string
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Source {i}] Document: {chunk['file_name']} | Page: {chunk['page']}\n"
                f"{chunk['content']}"
            )
        context = "\n\n".join(context_parts)

        user_prompt = f"""Based ONLY on the following HR policy excerpts, answer the employee question.

POLICY EXCERPTS:
{context}

EMPLOYEE QUESTION:
{question}

Remember:
- Answer strictly from the excerpts above.
- If the information is missing, say: "I could not find this information in the available HR policies."
- Cite the document name and page number.
"""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)

            # Handle different response formats from Gemini
            content = response.content

            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif hasattr(part, "text"):
                        text_parts.append(part.text)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                content = " ".join(text_parts)

            return str(content).strip()

        except Exception as e:
            return f"Sorry, I encountered an error while generating the answer: {str(e)}"

    def calculate_confidence(self, retrieved_chunks: List[Dict]) -> float:
        """Calculate confidence based on retrieval similarity scores."""
        if not retrieved_chunks:
            return 0.0

        scores = [c["score"] for c in retrieved_chunks]
        best = max(scores)
        avg_top3 = sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores))
        confidence = (0.6 * best + 0.4 * avg_top3) * 100
        return min(100.0, max(0.0, confidence))

    def ask(self, question: str) -> Dict:
        """Full RAG pipeline."""
        if not self._initialized:
            return {
                "answer": "System not initialized. Please check configuration.",
                "sources": [],
                "confidence": 0.0,
                "confidence_label": "Low Confidence (Below 60%)",
            }

        retrieved = self.retrieve(question, k=5)
        answer = self.generate_answer(question, retrieved)
        confidence = self.calculate_confidence(retrieved)

        if confidence >= 80:
            label = "High Confidence (80–100%)"
        elif confidence >= 60:
            label = "Medium Confidence (60–79%)"
        else:
            label = "Low Confidence (Below 60%)"

        # Deduplicate sources
        unique_sources = []
        seen = set()
        for c in retrieved:
            key = (c["file_name"], c["page"])
            if key not in seen:
                seen.add(key)
                unique_sources.append({
                    "file_name": c["file_name"],
                    "page": c["page"],
                })

        return {
            "answer": answer,
            "sources": unique_sources,
            "confidence": confidence,
            "confidence_label": label,
            "raw_chunks": retrieved,
        }


# Singleton instance used by the Streamlit app
rag_engine = HRPolicyRAG()