import os
from langchain_community.document_loaders import GitHubRepositoryLoader
loader = GitHubRepositoryLoader(
    clone_url="https://github.com/hwchase17/langchain",
    branch="main",
    file_filter=lambda file_path: file_path.endswith((".py", ".md"))
)
docs = loader.load()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(chunks, embeddings)
from langchain.chat_models import init_chat_model
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful coding assistant"),
    ("user", "Context:\n{context}\n\nQuestion: {question}")
])

def ask_question(question, retiever):
    docs = retiever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    filled_prompt = prompt.invoke({"context": context, "question": question})
    response = model.invoke(filled_prompt)
    return response.content

#Gradio UI
import gradio as gr
def qa_interface(repo_url, question):
    loader = GitHubRepositoryLoader(clone_url=repo_url, branch="main", file_filter=lambda x: x.endswith((".py", ".md")))
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return ask_question(question, vectorstore.as_retriever())

gr.Interface(
    fn=qa_interface,
    inputs=["text", "text"],
    outputs="text",
    title="🔍 Chat Your GitHub Repo",
    description="Enter a GitHub repo URL and ask questions about its code or docs!"
).launch()