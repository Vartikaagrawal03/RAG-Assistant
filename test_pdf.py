from langchain_community.document_loaders import PyPDFLoader
import os

pdf_path = "LAB MANUAL 5-8 Student.pdf"

if not os.path.exists(pdf_path):
    print("File not found! Make sure the PDF is in your project folder")
else:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Total pages: {len(docs)}")
    for i, doc in enumerate(docs[:5]):
        print(f"\n--- Page {i} ---")
        print(doc.page_content[:300])