import streamlit as st

from src.retriever import retrieve
from src.chatbot import generate_answer


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="PiezoGPT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(0, 210, 255, 0.08),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 15%,
                rgba(130, 80, 255, 0.08),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #060a14 0%,
                #0a1020 45%,
                #080d19 100%
            );

        color: #e8eefc;
    }


    /* ========================================================
       SUBTLE TECHNICAL GRID
       ======================================================== */

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;

        background-image:
            linear-gradient(
                rgba(255,255,255,0.018) 1px,
                transparent 1px
            ),
            linear-gradient(
                90deg,
                rgba(255,255,255,0.018) 1px,
                transparent 1px
            );

        background-size: 45px 45px;

        pointer-events: none;

        z-index: 0;
    }


    /* ========================================================
       MAIN CONTAINER
       ======================================================== */

    .block-container {
        max-width: 1100px;

        padding-top: 2.2rem;
        padding-bottom: 5rem;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        text-align: center;

        padding: 28px 20px 20px 20px;

        margin-bottom: 18px;
    }


    .hero-badge {
        display: inline-flex;

        align-items: center;
        gap: 8px;

        padding: 7px 14px;

        border-radius: 999px;

        background: rgba(0, 220, 255, 0.07);

        border: 1px solid rgba(0, 220, 255, 0.25);

        color: #71e8ff;

        font-size: 0.76rem;

        font-weight: 600;

        letter-spacing: 0.04em;

        margin-bottom: 18px;

        box-shadow:
            0 0 20px rgba(0, 220, 255, 0.08);
    }


    .hero-title {
        font-size: 4rem;

        font-weight: 850;

        letter-spacing: -0.055em;

        line-height: 1;

        margin-bottom: 12px;

        background:
            linear-gradient(
                100deg,
                #ffffff 10%,
                #7cecff 48%,
                #a98bff 90%
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;

        background-clip: text;
    }


    .hero-subtitle {
        color: #b5c2d9;

        font-size: 1.08rem;

        font-weight: 500;

        margin-bottom: 9px;
    }


    .hero-description {
        max-width: 680px;

        margin: auto;

        color: #75839d;

        font-size: 0.9rem;

        line-height: 1.65;
    }


    /* ========================================================
       ELECTRIC FIELD LINE
       ======================================================== */

    .field-line {
        width: 170px;

        height: 2px;

        margin: 22px auto 0 auto;

        background:
            linear-gradient(
                90deg,
                transparent,
                #36dfff,
                #9274ff,
                transparent
            );

        box-shadow:
            0 0 12px rgba(54, 223, 255, 0.45);
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .status-row {
        display: flex;

        justify-content: center;

        margin: 5px 0 28px 0;
    }


    .status {
        display: inline-flex;

        align-items: center;

        gap: 8px;

        padding: 7px 14px;

        border-radius: 999px;

        background: rgba(38, 190, 110, 0.07);

        border: 1px solid rgba(73, 220, 137, 0.25);

        color: #76e3a1;

        font-size: 0.76rem;

        font-weight: 600;
    }


    .status-dot {
        width: 7px;

        height: 7px;

        border-radius: 50%;

        background: #4fe28c;

        box-shadow:
            0 0 9px rgba(79, 226, 140, 0.9);
    }


    /* ========================================================
       FEATURE CARDS
       ======================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background:
            linear-gradient(
                145deg,
                rgba(18, 27, 49, 0.94),
                rgba(10, 17, 32, 0.92)
            );

        border: 1px solid rgba(120, 145, 190, 0.15);

        border-radius: 16px;

        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.18);

        transition:
            transform 0.2s ease,
            border-color 0.2s ease,
            box-shadow 0.2s ease;
    }


    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);

        border-color:
            rgba(74, 220, 255, 0.3);

        box-shadow:
            0 15px 40px rgba(0, 0, 0, 0.28),
            0 0 25px rgba(30, 190, 255, 0.05);
    }


    /* ========================================================
       FEATURE TEXT
       ======================================================== */

    .feature-icon {
        font-size: 1.45rem;

        margin-bottom: 4px;
    }


    .feature-heading {
        color: #f2f6ff;

        font-size: 0.94rem;

        font-weight: 700;

        margin-bottom: 5px;
    }


    .feature-text {
        color: #7887a2;

        font-size: 0.76rem;

        line-height: 1.5;
    }


    /* ========================================================
       SECTION LABEL
       ======================================================== */

    .section-label {
        color: #61708b;

        font-size: 0.7rem;

        font-weight: 700;

        text-transform: uppercase;

        letter-spacing: 0.14em;

        margin-top: 28px;

        margin-bottom: 10px;
    }


    /* ========================================================
       CHAT
       ======================================================== */

    [data-testid="stChatMessage"] {
        background: rgba(12, 19, 35, 0.72);

        border: 1px solid rgba(120, 145, 190, 0.08);

        border-radius: 16px;

        padding: 8px 14px;

        margin-bottom: 8px;
    }


    /* ========================================================
       CHAT INPUT
       ======================================================== */

    [data-testid="stChatInput"] {
        border-radius: 16px;
    }


    [data-testid="stChatInput"] textarea {
        background: #0d1526 !important;

        color: #edf5ff !important;

        border: 1px solid #263652 !important;

        border-radius: 14px !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #070c17 0%,
                #080e1b 100%
            );

        border-right: 1px solid rgba(100, 130, 180, 0.08);
    }


    section[data-testid="stSidebar"] h1 {
        color: #eef6ff;
    }


    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #dce8fb;
    }


    section[data-testid="stSidebar"] p {
        color: #7e8da8;
    }


    /* ========================================================
       METRIC
       ======================================================== */

    [data-testid="stMetric"] {
        background: rgba(15, 24, 43, 0.7);

        border: 1px solid rgba(100, 130, 180, 0.1);

        padding: 12px;

        border-radius: 12px;
    }


    [data-testid="stMetricValue"] {
        color: #72e8ff;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton button {
        border-radius: 10px;

        border: 1px solid #263752;

        background: #101a2e;

        color: #b9c8df;

        transition: all 0.2s ease;
    }


    .stButton button:hover {
        border-color: #35dfff;

        color: #ffffff;

        background: #13223a;

        box-shadow:
            0 0 18px rgba(53, 223, 255, 0.08);
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color: rgba(120, 145, 190, 0.1);
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 700px) {

        .hero-title {
            font-size: 2.7rem;
        }

        .hero-subtitle {
            font-size: 0.92rem;
        }

        .hero-description {
            font-size: 0.82rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">
        <div class="hero-badge">
            ⚡ PIEZOELECTRIC KNOWLEDGE SYSTEM
        </div>

        <div class="hero-title">
            PiezoGPT
        </div>

        <div class="hero-subtitle">
            Linear Piezoelectric Plate Vibrations
        </div>

        <div class="hero-description">
            A domain-specific AI assistant grounded exclusively
            in H. F. Tiersten's textbook, combining semantic
            retrieval with Gemini-powered technical reasoning.
        </div>

        <div class="field-line"></div>
    </div>
    """
)


# ============================================================
# STATUS
# ============================================================

st.html(
    """
    <div class="status-row">
        <div class="status">
            <span class="status-dot"></span>
            Knowledge base online
        </div>
    </div>
    """
)


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3 = st.columns(3, gap="medium")


with col1:

    with st.container(border=True):

        st.markdown(
            '<div class="feature-icon">📚</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-heading">'
            'Textbook Grounded'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-text">'
            "Answers are grounded in Tiersten's "
            "Linear Piezoelectric Plate Vibrations."
            '</div>',
            unsafe_allow_html=True,
        )


with col2:

    with st.container(border=True):

        st.markdown(
            '<div class="feature-icon">⚡</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-heading">'
            'Semantic Retrieval'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-text">'
            "Technical passages are retrieved and "
            "reranked before generation."
            '</div>',
            unsafe_allow_html=True,
        )


with col3:

    with st.container(border=True):

        st.markdown(
            '<div class="feature-icon">🧠</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-heading">'
            'Gemini Intelligence'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-text">'
            "Grounded technical responses with "
            "textbook page citations."
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# ASK SECTION
# ============================================================

st.markdown(
    '<div class="section-label">Ask the textbook</div>',
    unsafe_allow_html=True,
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================

question = st.chat_input(
    "Ask about piezoelectricity, equations, vibrations..."
)


# ============================================================
# PROCESS QUESTION
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
    # Assistant
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching Tiersten's textbook..."
        ):

            try:

                # =================================================
                # Retrieval
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
                # Generate
                # =================================================

                else:

                    answer = generate_answer(
                        question,
                        results
                    )


                # =================================================
                # Display
                # =================================================

                st.markdown(
                    answer
                )


                # =================================================
                # Save
                # =================================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )


            except Exception as e:

                st.error(
                    "Sorry, I encountered an error while "
                    "processing your question."
                )

                st.exception(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div style="
            font-size: 1.45rem;
            font-weight: 800;
            color: #eef6ff;
            margin-bottom: 4px;
        ">
            ⚡ PiezoGPT
        </div>

        <div style="
            font-size: 0.76rem;
            color: #6f7e98;
            line-height: 1.5;
        ">
            Scientific AI assistant for
            Linear Piezoelectric Plate Vibrations.
        </div>
        """
    )


    st.divider()


    # --------------------------------------------------------
    # System
    # --------------------------------------------------------

    st.subheader("⚙️ System")


    st.markdown(
        """
        **Retrieval**

        Semantic + technical reranking

        **Knowledge**

        H. F. Tiersten

        **Generation**

        Gemini
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
        "Indexed technical content from the textbook."
    )


    st.divider()


    # --------------------------------------------------------
    # How It Works
    # --------------------------------------------------------

    st.subheader("🔬 Pipeline")

    st.markdown(
        """
        **01** — Ask

        **02** — Retrieve

        **03** — Rerank

        **04** — Generate

        **05** — Cite
        """
    )


    st.divider()


    # --------------------------------------------------------
    # Clear Chat
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


    st.divider()


    st.caption(
        "PiezoGPT • Scientific RAG Assistant"
    )