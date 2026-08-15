import streamlit as st
import requests
import os

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="3GPP Telecom RAG Assistant",
    page_icon="📡",
    layout="wide"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        margin-bottom: 25px;
    }

    .source-box {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">3GPP Telecom RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    A Retrieval-Augmented Generation system for intelligent
    question answering over 3GPP telecommunications standards
    and technical specifications, with responses grounded in
    indexed technical documentation.
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Backend URL
# --------------------------------------------------

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    
 "https://threegpp-rag-chatbot-nl2c.onrender.com/chat"
)


# --------------------------------------------------
# Question input
# --------------------------------------------------

question = st.text_input(
    "Ask a question",
    placeholder="Example: What is the role of the AMF?"
)


# --------------------------------------------------
# Ask button
# --------------------------------------------------

if st.button("Ask", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching 3GPP specifications..."):

            try:

                response = requests.post(
                    BACKEND_URL,
                    json={
                        "question": question
                    },
                    timeout=120
                )

                response.raise_for_status()

                result = response.json()


                # ------------------------------------------
                # Answer
                # ------------------------------------------

                st.subheader("Answer")

                st.write(result.get("answer", ""))


                # ------------------------------------------
                # Sources
                # ------------------------------------------

                sources = result.get("sources", [])

                if sources:

                    st.subheader("📚 Sources")

                    displayed_sources = set()

                    for source in sources:

                        document = source.get(
                            "document",
                            "Unknown"
                        )

                        page = source.get(
                            "page",
                            "Unknown"
                        )

                        key = (document, page)

                        if key in displayed_sources:
                            continue

                        displayed_sources.add(key)

                        st.markdown(
                            f"""
                            <div class="source-box">
                            <b>Document:</b> {document}<br>
                            <b>Page:</b> {page}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                else:

                    st.info(
                        "No supporting 3GPP source was found."
                    )


            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI backend. "
                    "Make sure the backend is running on port 8000."
                )


            except requests.exceptions.Timeout:

                st.error(
                    "The request took too long. "
                    "Please try again."
                )


            except requests.exceptions.RequestException as e:

                st.error(
                    f"Backend request failed: {e}"
                )