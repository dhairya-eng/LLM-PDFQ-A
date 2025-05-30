import os
import streamlit as st
import tempfile
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate

# Set up Gemini API key or you llm api key 
# if not os.environ.get("GOOGLE_API_KEY"):
#     os.environ["GOOGLE_API_KEY"] = st.text_input("Enter your Google Gemini API Key", type="password")
#     st.stop()

# Initialize Gemini LLM and Embeddings
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the following PDF content to answer the user's question."),
    ("user", "Context:\n{context}\n\nQuestion: {question}")
])

# PDF text extraction
def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    doc = fitz.open(tmp_path)
    text = "\n".join(page.get_text() for page in doc)
    return text

# Vector indexing
def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    return FAISS.from_documents(docs, embeddings)

# Ask question with RAG
def ask_question(question, vectorstore):
    retriever = vectorstore.as_retriever()
    matches = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in matches])
    filled_prompt = prompt.invoke({"context": context, "question": question})
    response = model.invoke(filled_prompt)
    return response.content

# Streamlit UI
st.title("📄 RAG PDF Q&A with Gemini")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded. Extracting and indexing content...")
    text = extract_text_from_pdf(uploaded_file)
    vectorstore = create_vectorstore(text)
    st.success("Ready! Ask your question below:")

    question = st.text_input("Ask a question about the PDF")

    if question and vectorstore:
        with st.spinner("Searching and thinking..."):
            answer = ask_question(question, vectorstore)
        st.markdown("### Answer:")
        st.write(answer)