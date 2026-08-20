import streamlit as st

from src.retriever import retrieve
from src.chatbot import generate_answer


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="PiezoGPT",
    page_icon="⚡",
    layout="centered",
)


# ============================================================
# Header
# ============================================================

st.title("⚡ PiezoGPT")

st.caption(
    "AI assistant for *Linear Piezoelectric Plate Vibrations* "
    "by H. F. Tiersten"
)

st.markdown(
    """
    Ask questions about the textbook and PiezoGPT will retrieve
    relevant passages from the knowledge base before generating
    an answer using Gemini.
    """
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("About PiezoGPT")

    st.markdown(
        """
        **PiezoGPT** is a Retrieval-Augmented Generation (RAG)
        application built around:

        - 📚 H. F. Tiersten's textbook
        - 🔎 Semantic retrieval
        - 🧠 Sentence Transformers
        - 🗄️ ChromaDB
        - 🤖 Gemini
        """
    )

    st.divider()

    st.markdown(
        "**Knowledge Base**"
    )

    st.write("539 textbook chunks")

    st.divider()

    st.caption(
        "Answers are grounded in the retrieved textbook content."
    )


# ============================================================
# Chat History
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# Display Previous Messages
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# User Input
# ============================================================

question = st.chat_input(
    "Ask a question about piezoelectricity..."
)


if question:

    # --------------------------------------------------------
    # Store user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Retrieving textbook content and generating answer..."
        ):

            try:

                # ------------------------------------------------
                # Retrieve relevant textbook chunks
                # ------------------------------------------------

                results = retrieve(question)

                # ------------------------------------------------
                # Handle no retrieval results
                # ------------------------------------------------

                if not results:

                    answer = (
                        "The provided textbook context does not "
                        "contain enough information to answer this "
                        "confidently."
                    )

                else:

                    # ------------------------------------------------
                    # Generate answer
                    #
                    # IMPORTANT:
                    # Pass the original retrieval results.
                    # generate_answer() calls build_context()
                    # internally.
                    # ------------------------------------------------

                    answer = generate_answer(
                        question,
                        results
                    )

                # ------------------------------------------------
                # Display answer
                # ------------------------------------------------

                st.markdown(answer)

                # ------------------------------------------------
                # Store assistant response
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as e:

                error_message = (
                    "Sorry, I encountered an error while "
                    "processing your question."
                )

                st.error(error_message)

                st.exception(e)