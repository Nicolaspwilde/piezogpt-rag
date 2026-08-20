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
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       Main Application
       ======================================================== */

    .stApp {
        background-color: #0b1020;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ========================================================
       Header
       ======================================================== */

    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
        color: #ffffff;
        letter-spacing: -1px;
    }

    .main-subtitle {
        text-align: center;
        color: #a8b3c7;
        font-size: 1.05rem;
        margin-bottom: 0.6rem;
    }

    .main-description {
        text-align: center;
        color: #7f8ba3;
        max-width: 650px;
        margin: 0 auto;
        line-height: 1.6;
        font-size: 0.9rem;
    }


    /* ========================================================
       Status
       ======================================================== */

    .status-wrapper {
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }

    .status {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        background-color: #10251a;
        border: 1px solid #245c3a;
        color: #7ed69a;
        font-size: 0.78rem;
    }


    /* ========================================================
       Sidebar
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #080d19;
    }


    /* ========================================================
       Chat
       ======================================================== */

    [data-testid="stChatMessage"] {
        border-radius: 12px;
    }


    /* ========================================================
       Chat Input
       ======================================================== */

    [data-testid="stChatInput"] {
        border-radius: 12px;
    }


    /* ========================================================
       Mobile
       ======================================================== */

    @media (max-width: 700px) {

        .main-title {
            font-size: 2.4rem;
        }

        .main-subtitle {
            font-size: 0.95rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">⚡ PiezoGPT</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-subtitle">'
    'AI assistant for <i>Linear Piezoelectric Plate Vibrations</i>'
    ' by H. F. Tiersten'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-description">'
    "Ask questions about the textbook. "
    "PiezoGPT retrieves relevant passages from the "
    "knowledge base and generates grounded answers using Gemini."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="status-wrapper">
        <span class="status">
            ● Knowledge base online
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Feature Cards
# ============================================================
#
# IMPORTANT:
# These cards intentionally use native Streamlit components.
# Do NOT use <div> HTML here.
#

col1, col2, col3 = st.columns(3)


with col1:

    with st.container(border=True):

        st.markdown("### 📚")

        st.markdown(
            "**Textbook Grounded**"
        )

        st.caption(
            "Answers are grounded in Tiersten's textbook."
        )


with col2:

    with st.container(border=True):

        st.markdown("### 🔎")

        st.markdown(
            "**Smart Retrieval**"
        )

        st.caption(
            "Relevant technical passages are retrieved "
            "from the knowledge base."
        )


with col3:

    with st.container(border=True):

        st.markdown("### 🤖")

        st.markdown(
            "**Gemini Powered**"
        )

        st.caption(
            "Gemini generates concise, grounded answers."
        )


st.markdown("")


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("⚡ PiezoGPT")

    st.caption(
        "Domain-specific RAG assistant for "
        "Linear Piezoelectric Plate Vibrations."
    )

    st.divider()


    # --------------------------------------------------------
    # How It Works
    # --------------------------------------------------------

    st.subheader("🧠 How it works")

    st.markdown(
        """
        **1. Ask**

        Ask a question about the textbook.

        **2. Retrieve**

        Relevant textbook passages are retrieved.

        **3. Generate**

        Gemini generates an answer from the retrieved content.

        **4. Cite**

        Relevant textbook page numbers are provided.
        """
    )

    st.divider()


    # --------------------------------------------------------
    # Knowledge Base
    # --------------------------------------------------------

    st.subheader("📚 Knowledge Base")

    st.metric(
        label="Textbook chunks",
        value="539",
    )

    st.caption(
        "Answers are grounded in the indexed textbook."
    )

    st.divider()


    # --------------------------------------------------------
    # Clear Chat
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        "PiezoGPT • RAG-based technical assistant"
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

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# User Input
# ============================================================

question = st.chat_input(
    "Ask something about piezoelectricity..."
)


# ============================================================
# Process Question
# ============================================================

if question:

    # --------------------------------------------------------
    # Store User Message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # Assistant Response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the textbook..."
        ):

            try:

                # =================================================
                # Retrieve Relevant Textbook Chunks
                # =================================================

                results = retrieve(
                    question
                )


                # =================================================
                # No Results
                # =================================================

                if not results:

                    answer = (
                        "The provided textbook context does not "
                        "contain enough information to answer this "
                        "confidently."
                    )


                # =================================================
                # Generate Answer
                # =================================================

                else:

                    # IMPORTANT:
                    #
                    # Pass the ORIGINAL list returned by retrieve().
                    #
                    # generate_answer() calls build_context()
                    # internally.
                    #
                    # Do NOT convert results into a string.

                    answer = generate_answer(
                        question,
                        results
                    )


                # =================================================
                # Display Answer
                # =================================================

                st.markdown(
                    answer
                )


                # =================================================
                # Store Assistant Response
                # =================================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )


            # =====================================================
            # Error Handling
            # =====================================================

            except Exception as e:

                st.error(
                    "Sorry, I encountered an error while "
                    "processing your question."
                )

                st.exception(e)