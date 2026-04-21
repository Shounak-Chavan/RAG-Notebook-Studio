import os
import shutil

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embedding = None
vectorstore = None
chat_history = []


def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text.strip()


def split_text(text):
    docs = [Document(page_content=text, metadata={"source": "uploaded_pdf"})]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


def create_embeddings(chunks):
    global embedding
    embedding = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
    )
    sample_text = "Hi This is a sample text to test the embedding generation."
    _ = embedding.embed_query(sample_text)
    return embedding


def build_faiss_index(chunks):
    global vectorstore
    emb = create_embeddings(chunks)
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=emb,
    )

    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    vectorstore.save_local("faiss_index")
    return vectorstore


def query_faiss(question):
    if vectorstore is None:
        return []
    return vectorstore.similarity_search(question, k=3)


def call_groq_llm(context, question, chat_history):
    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=200,
        groq_api_key=GROQ_API_KEY,
    )

    history_text = "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in chat_history]
    )

    prompt = f"""Answer the question based only on the context below and conversation history.

Context:
{context}

Conversation:
{history_text}

Question:
{question}

If the answer is not in context, say you don't know."""

    response = model.invoke(prompt)
    return response.content


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global chat_history

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF is allowed"}), 400

    text = load_pdf(file)
    if not text:
        return jsonify({"error": "Could not extract text from PDF"}), 400

    chunks = split_text(text)
    build_faiss_index(chunks)

    chat_history = []

    return jsonify(
        {
            "message": "PDF uploaded and indexed successfully",
            "chunks": len(chunks),
            "vectors": vectorstore.index.ntotal if vectorstore else 0,
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    global chat_history

    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY not set"}), 500

    if vectorstore is None:
        return jsonify({"error": "Upload a PDF first"}), 400

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    docs = query_faiss(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    recent_history = chat_history[-5:]
    answer = call_groq_llm(context, question, recent_history)

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
    chat_history = chat_history[-10:]

    return jsonify({"answer": answer})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
