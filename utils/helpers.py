"""
Utility helper functions for the HR Policy Assistant.
"""

import os
from datetime import datetime
from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


def ensure_directories():
    """Create required directories if they do not exist."""
    root = get_project_root()
    (root / "data").mkdir(exist_ok=True)
    (root / "chroma_db").mkdir(exist_ok=True)
    (root / "analytics").mkdir(exist_ok=True)


def log_interaction(question: str, answer: str, confidence: float, sources: list):
    """
    Append a question-answer interaction to analytics/questions.csv.
    """
    ensure_directories()
    csv_path = get_project_root() / "analytics" / "questions.csv"

    source_str = "; ".join(
        [f"{s.get('file_name', 'Unknown')} (Page {s.get('page', '?')})" for s in sources]
    )

    row = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "confidence": round(confidence, 2),
        "source_documents": source_str,
    }

    df = pd.DataFrame([row])

    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, mode="w", header=True, index=False)


def load_analytics() -> pd.DataFrame:
    """Load analytics data. Returns empty DataFrame if file does not exist."""
    csv_path = get_project_root() / "analytics" / "questions.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=["timestamp", "question", "answer", "confidence", "source_documents"])