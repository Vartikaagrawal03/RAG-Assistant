import streamlit as st
import tempfile
import os
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from rag_pipeline import process_documents

st.set_page_config(
    page_title="RAG Documentation Assistant",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #f8fafc; }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    .brand-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.5rem 0 1rem 0;
    }
    .brand-icon {
        width: 38px;
        height: 38px;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }
    .brand-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1.2;
    }
    .brand-tagline {
        font-size: 0.7rem;
        color: #94a3b8;
        font-weight: 400;
    }

    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1rem 0 0.5rem 0;
    }

    .file-chip {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 5px 10px;
        font-size: 0.78rem;
        color: #475569;
        margin: 3px 0;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .stat-row {
        display: flex;
        gap: 8px;
        margin: 8px 0;
    }
    .stat-card {
        flex: 1;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 8px;
        text-align: center;
    }
    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #3b82f6;
        line-height: 1;
    }
    .stat-label {
        font-size: 0.68rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 3px;
    }

    .page-header {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .header-icon {
        width: 56px;
        height: 56px;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        flex-shrink: 0;
    }
    .header-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1.2;
        margin: 0;
    }
    .header-subtitle {
        font-size: 0.88rem;
        color: #64748b;
        margin: 4px 0 0 0;
    }
    .header-badges {
        display: flex;
        gap: 8px;
        margin-top: 8px;
        flex-wrap: wrap;
    }
    .badge {
        background: #eff6ff;
        color: #3b82f6;
        border: 1px solid #bfdbfe;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 500;
    }

    .summary-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 1.2rem;
        border-left: 4px solid #10b981;
    }
    .summary-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #10b981;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 10px;
    }

    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 4px 8px;
        margin: 6px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #3b82f6;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.8rem;
        color: #475569;
    }
    .source-header {
        font-weight: 600;
        color: #3b82f6;
        font-size: 0.78rem;
        margin-bottom: 4px;
    }

    .welcome-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 50px 40px;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 24px;
        text-align: left;
    }
    .feature-item {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px;
    }
    .feature-icon { font-size: 1.3rem; margin-bottom: 6px; }
    .feature-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 2px;
    }
    .feature-desc {
        font-size: 0.74rem;
        color: #94a3b8;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []
if "summary" not in st.session_state:
    st.session_state.summary = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None

with st.sidebar:
    st.markdown("""
    <div class="brand-logo">
        <div class="brand-icon">📄</div>
        <div>
            <div class="brand-name">RAG Assistant</div>
            <div class="brand-tagline">Groq · LangChain · FAISS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for f in uploaded_files:
            ext = f.name.split(".")[-1].upper()
            st.markdown(f"""
            <div class="file-chip">
                <span>📎</span>
                <span>{f.name}</span>
                <span style='margin-left:auto; background:#e0f2fe; color:#0284c7;
                border-radius:4px; padding:1px 6px; font-size:0.65rem;'>{ext}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown(" ")
        if st.button("⚡ Process Documents", type="primary", use_container_width=True):
            progress = st.progress(0, text="Loading documents...")
            tmp_paths = []
            for uploaded_file in uploaded_files:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_paths.append(tmp.name)

            progress.progress(25, text="Splitting into chunks...")
            from rag_pipeline import load_documents, split_documents, create_vector_store, create_qa_chain, generate_summary

            documents = load_documents(tmp_paths)
            chunks = split_documents(documents)

            progress.progress(50, text="Creating vector store...")
            vector_store = create_vector_store(chunks)

            progress.progress(75, text="Building AI chain...")
            chain, retriever, llm = create_qa_chain(vector_store)

            progress.progress(90, text="Generating summary...")
            summary = generate_summary(documents, llm)

            total_pages = 0
            for doc in documents:
                page = doc.metadata.get("page", None)
                if page is not None:
                    total_pages = max(total_pages, int(page) + 1)
                else:
                    total_pages += 1

            st.session_state.qa_chain = chain
            st.session_state.retriever = retriever
            st.session_state.summary = summary
            st.session_state.doc_stats = {
                "files": [f.name for f in uploaded_files],
                "chunks": len(chunks),
                "docs": total_pages
            }
            progress.progress(100, text="Done!")
            st.success("✅ Ready to answer questions!")

    if st.session_state.doc_stats:
        st.markdown('<div class="section-label">Stats</div>', unsafe_allow_html=True)
        stats = st.session_state.doc_stats
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-number">{len(stats['files'])}</div>
                <div class="stat-label">Files</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['docs']}</div>
                <div class="stat-label">Pages</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['chunks']}</div>
                <div class="stat-label">Chunks</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.qa_chain:
        st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)
        if st.button("🔄 New Session", use_container_width=True):
            for key in ["qa_chain", "retriever", "summary", "doc_stats"]:
                st.session_state[key] = None
            st.session_state.chat_history = []
            st.session_state.langchain_history = []
            st.rerun()

    if st.session_state.chat_history:
        chat_export = ""
        for role, message, _ in st.session_state.chat_history:
            chat_export += f"{role}:\n{message}\n\n"
        st.download_button(
            label="💾 Export Chat",
            data=chat_export,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.langchain_history = []
            st.rerun()

# Main area
st.markdown("""
<div class="page-header">
    <div class="header-icon">📄</div>
    <div>
        <p class="header-title">Documentation Assistant</p>
        <p class="header-subtitle">Ask questions about your documents — get answers with source attribution</p>
        <div class="header-badges">
            <span class="badge">📚 Multi-document</span>
            <span class="badge">📌 Source attribution</span>
            <span class="badge">🧠 Conversational memory</span>
            <span class="badge">⚡ Groq Llama 3</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.summary:
    st.markdown(f"""
    <div class="summary-card">
        <div class="summary-title">📋 Document Summary</div>
        {st.session_state.summary}
    </div>
    """, unsafe_allow_html=True)

if st.session_state.qa_chain:
    for role, message, timestamp in st.session_state.chat_history:
        if role == "You":
            with st.chat_message("user"):
                st.write(message)
                st.caption(f"🕐 {timestamp}")
        else:
            with st.chat_message("assistant"):
                st.write(message)

    user_question = st.chat_input("Ask something about your documents...")

    if user_question:
        timestamp = datetime.now().strftime("%H:%M:%S")

        with st.spinner("Thinking..."):
            answer = st.session_state.qa_chain.invoke({
                "question": user_question,
                "chat_history": st.session_state.langchain_history
            })
            source_docs = st.session_state.retriever.invoke(user_question)

        st.session_state.langchain_history.append(HumanMessage(content=user_question))
        st.session_state.langchain_history.append(AIMessage(content=answer))
        st.session_state.chat_history.append(("You", user_question, timestamp))
        st.session_state.chat_history.append(("Assistant", answer, timestamp))
        st.rerun()

else:
    st.markdown("""
    <div class="welcome-card">
        <div style='font-size:3rem; margin-bottom:12px'>📄</div>
        <div style='font-size:1.4rem; font-weight:700; color:#1e293b;'>
            Welcome to RAG Assistant
        </div>
        <div style='color:#64748b; font-size:0.9rem; margin-top:6px;'>
            Upload your documents in the sidebar and start asking questions
        </div>
        <div class="feature-grid">
            <div class="feature-item">
                <div class="feature-icon">📚</div>
                <div class="feature-title">Multi-Document</div>
                <div class="feature-desc">Upload and query multiple files at once</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">📌</div>
                <div class="feature-title">Source Attribution</div>
                <div class="feature-desc">Know exactly where each answer comes from</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">Memory</div>
                <div class="feature-desc">Ask follow-up questions naturally</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Fast Inference</div>
                <div class="feature-desc">Powered by Groq's ultra-fast API</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">📋</div>
                <div class="feature-title">Auto Summary</div>
                <div class="feature-desc">Get a summary as soon as you upload</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">💾</div>
                <div class="feature-title">Export Chat</div>
                <div class="feature-desc">Download your Q&A session anytime</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)