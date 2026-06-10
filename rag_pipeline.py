import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

def load_documents(file_paths):
    all_documents = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
          from langchain_community.document_loaders import PDFPlumberLoader
          loader = PDFPlumberLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = os.path.basename(file_path)
        all_documents.extend(docs)
    return all_documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=60
)
    return splitter.split_documents(documents)

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def generate_summary(documents, llm):
    combined_text = " ".join([doc.page_content for doc in documents[:5]])
    combined_text = combined_text[:3000]
    prompt = f"Summarize the following document in 5 clear bullet points:\n\n{combined_text}"
    summary = llm.invoke(prompt)
    return summary.content

def format_docs_with_sources(docs):
    results = []
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "N/A")
        results.append(f"[Source: {source} | Page: {page}]\n{doc.page_content}")
    return "\n\n".join(results)

def create_qa_chain(vector_store):
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the question based only on the context below.
If you don't know the answer, say "I don't have enough information to answer that."
Always mention which source/page your answer comes from.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    def get_context(input_dict):
        docs = retriever.invoke(input_dict["question"])
        return format_docs_with_sources(docs)

    chain = (
        {
            "context": lambda x: get_context(x),
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever, llm

def process_documents(file_paths):
    print("Loading documents...")
    documents = load_documents(file_paths)

    print("Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Creating vector store...")
    vector_store = create_vector_store(chunks)

    print("Building QA chain...")
    chain, retriever, llm = create_qa_chain(vector_store)

    print("Generating summary...")
    summary = generate_summary(documents, llm)

    total_pages = 0
    for doc in documents:
        page = doc.metadata.get("page", None)
        if page is not None:
            total_pages = max(total_pages, int(page) + 1)
        else:
            total_pages += 1

    print("Ready!")
    return chain, retriever, summary, len(chunks), total_pages