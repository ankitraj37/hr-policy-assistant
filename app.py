"""
app.py
------
Streamlit frontend for the HR Policy Assistant.
Employees only ask questions – no document upload UI.
"""

import streamlit as st
from pathlib import Path
import pandas as pd

from rag import rag_engine
from utils.helpers import log_interaction, load_analytics, ensure_directories

# Page config
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure folders exist
ensure_directories()

# ---------- Sidebar ----------
with st.sidebar:
    st.title("📋 HR Policy Assistant")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Chat", "Analytics"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Ask any question about company HR policies.")
    st.caption("Answers are grounded in official policy documents only.")


# ---------- Initialize RAG once ----------
@st.cache_resource
def init_rag():
    success, message = rag_engine.initialize()
    return success, message


success, init_message = init_rag()

if not success:
    st.error(init_message)
    st.stop()


# ---------- Chat Page ----------
if page == "Chat":
    st.title("HR Policy Assistant")
    st.markdown("Ask questions about leave, travel, remote work, notice period, reimbursements, and more.")

    # Session state for conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Sources & Confidence"):
                    for src in msg["sources"]:
                        st.write(f"• **{src['file_name']}** (Page {src['page']})")
                    st.write(f"**Confidence:** {msg.get('confidence_label', '')} — {msg.get('confidence', 0):.0f}%")

    # Chat input
    if question := st.chat_input("Type your HR policy question here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching policies and generating answer..."):
                result = rag_engine.ask(question)

            answer = result["answer"]
            sources = result["sources"]
            confidence = result["confidence"]
            confidence_label = result["confidence_label"]

            st.markdown(answer)

            with st.expander("📚 Sources & Confidence", expanded=True):
                if sources:
                    st.markdown("**Sources:**")
                    for src in sources:
                        st.markdown(f"- `{src['file_name']}` (Page {src['page']})")
                else:
                    st.markdown("_No specific sources retrieved._")

                st.markdown(f"**Confidence:** {confidence_label} — **{confidence:.0f}%**")

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "confidence": confidence,
                "confidence_label": confidence_label,
            })

            # Log for analytics
            try:
                log_interaction(question, answer, confidence, sources)
            except Exception as e:
                st.warning(f"Could not log analytics: {e}")


# ---------- Analytics Page ----------
elif page == "Analytics":
    st.title("📊 Analytics Dashboard")

    df = load_analytics()

    if df.empty:
        st.info("No questions have been asked yet. Start chatting to see analytics.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Questions Asked", len(df))

        with col2:
            avg_conf = df["confidence"].mean()
            st.metric("Average Confidence", f"{avg_conf:.1f}%")

        with col3:
            # Most referenced policy (simple heuristic)
            all_sources = " ".join(df["source_documents"].fillna("").astype(str))
            # Extract filenames
            from collections import Counter
            import re
            files = re.findall(r"([\w\-]+\.pdf)", all_sources)
            if files:
                most_common = Counter(files).most_common(1)[0]
                st.metric("Most Referenced Policy", most_common[0])
            else:
                st.metric("Most Referenced Policy", "N/A")

        st.markdown("---")
        st.subheader("Recent Questions")
        st.dataframe(
            df[["timestamp", "question", "confidence", "source_documents"]].sort_values(
                "timestamp", ascending=False
            ).head(20),
            use_container_width=True,
        )

        st.subheader("Most Common Questions")
        # Simple frequency of exact questions
        top_q = df["question"].value_counts().head(10)
        st.bar_chart(top_q)