import streamlit as st
import tempfile
import os
from rag_pipeline import process_document

st.set_page_config(page_title="RAG Documentation Assistant", page_icon="📄")
st.title("📄 RAG Documentation Assistant")
st.markdown("Upload a document and ask questions about it!")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "txt", "docx"]
)

if uploaded_file is not None and st.session_state.qa_chain is None:
    with st.spinner("Processing document... please wait..."):
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        st.session_state.qa_chain = process_document(tmp_path)
    st.success("Document ready! Ask your questions below.")

if st.session_state.qa_chain:
    user_question = st.chat_input("Ask something about your document...")

    if user_question:
        with st.spinner("Thinking..."):
            answer = st.session_state.qa_chain.invoke(user_question)

        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Assistant", answer))

    for role, message in st.session_state.chat_history:
        if role == "You":
            with st.chat_message("user"):
                st.write(message)
        else:
            with st.chat_message("assistant"):
                st.write(message)