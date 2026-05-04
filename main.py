import os
import re
import shutil

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USE_FAKE_MODELS = True

embedding_model = None
vectorstore = None
faiss_retriever = None
bm25_retriever = None
hybrid_retriever = None
reranker = None
chat_history = []
uploaded_text = ""
sections = {
    "authors": "",
    "references": "",
    "abstract": "",
}


class FakeCrossEncoder:
    """Lightweight no-op reranker for test mode."""

    def predict(self, pairs):
        return [0.0 for _ in pairs]


def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text.strip()


def detect_intent(question: str) -> str:
    """Detect the intent of a question."""
    question_lower = question.lower()
    
    authors_keywords = ["who made", "who wrote", "authors", "written by", "author names", "who authored"]
    if any(kw in question_lower for kw in authors_keywords):
        return "authors"
    
    references_keywords = [
        "reference",
        "references",
        "refrence",
        "refrences",
        "bibliography",
        "citations",
        "cite",
        "list of references",
        "show refs",
        "ref",
    ]
    if any(kw in question_lower for kw in references_keywords) or re.search(r"\breferenc|\brefrenc|\bbibliograph|\bcitat", question_lower):
        return "references"
    
    abstract_keywords = ["abstract", "summary", "overview"]
    if any(kw in question_lower for kw in abstract_keywords):
        return "abstract"
    
    return "general"


def extract_authors(text: str) -> str:
    """Extract author names from the first ~40 lines of the document."""
    lines = text.split("\n")[:80]
    authors = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        candidate = re.sub(r"^(?:Dr|Prof|Mr|Ms|Mrs)\.?\s+", "", line, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"\s+", " ", candidate)
        candidate_lower = candidate.lower()

        # Skip affiliation / section / email lines.
        if any(marker in candidate_lower for marker in ["dept", "department", "university", "college", "institute", "email", "@", "abstract", "introduction", "references", "pune, india", "india"]):
            continue

        # Skip obvious title/noise lines before the author block.
        if ":" in line:
            continue
        if any(keyword in candidate_lower for keyword in ["assettrack", "inventory", "management", "system", "paper", "document"]):
            continue

        # Keep only short name-like lines from the header region.
        tokens = [token.strip(".,") for token in candidate.split() if token.strip(".,")]
        if not (2 <= len(tokens) <= 6):
            continue
        if len(candidate) > 80:
            continue

        # Require at least two personal-name tokens and reject initials-only lines like "I. I".
        name_tokens = [
            token for token in tokens
            if re.fullmatch(r"[A-Z][a-z]{2,}(?:-[A-Z][a-z]{2,})?", token)
        ]
        if len(name_tokens) < 2:
            continue

        # Accept author lines that are actual names, with optional initials.
        valid = True
        for token in tokens:
            if token.lower() in {"and", "&"}:
                continue
            if re.fullmatch(r"[A-Z]\.", token):
                continue
            if re.fullmatch(r"[A-Z][a-z]{2,}(?:-[A-Z][a-z]{2,})?", token):
                continue
            valid = False
            break

        if valid:
            authors.append(candidate)
    
    if authors:
        # De-duplicate while preserving order.
        seen = set()
        unique_authors = []
        for author in authors:
            if author not in seen:
                seen.add(author)
                unique_authors.append(author)
        return "Authors:\n" + "\n".join(unique_authors[:10])
    return ""


def extract_abstract(text: str) -> str:
    """Extract abstract from text between 'Abstract' and 'Introduction'."""
    abstract_match = re.search(
        r"(?:^|\n)\s*abstract\s*(?:\n|$)(.*?)(?:\n\s*(?:introduction|i\.|keywords|index))",
        text,
        re.IGNORECASE | re.DOTALL
    )
    
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        abstract_text = re.sub(r"\s+", " ", abstract_text)
        return abstract_text[:1000]
    
    return ""


def extract_references(text):
    idx = text.lower().rfind("references")
    if idx == -1:
        return ""

    ref_text = text[idx:]
    pattern = r"\[\d+\].*?(?=\[\d+\]|$)"
    matches = re.findall(pattern, ref_text, re.DOTALL)

    cleaned = []
    for m in matches:
        cleaned.append(" ".join(m.split()))

    return "\n\n".join(cleaned)


def split_text(text):
    docs = [Document(page_content=text, metadata={"source": "uploaded_pdf"})]
    emb = create_embeddings()

    chunker = SemanticChunker(
        embeddings=emb,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )
    sem_chunks = chunker.split_documents(docs)

    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "। ", ". ", " ", ""],
    )

    final_chunks = []
    for chunk in sem_chunks:
        if len(chunk.page_content) > 1000:
            final_chunks.extend(rec_splitter.split_documents([chunk]))
        else:
            final_chunks.append(chunk)

    filtered_chunks = [
        chunk for chunk in final_chunks if len(chunk.page_content.strip()) > 120
    ]
    return filtered_chunks


def create_embeddings():
    global embedding_model
    if embedding_model is None:
        if USE_FAKE_MODELS:
            embedding_model = FakeEmbeddings(size=384)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings

            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    return embedding_model


def build_faiss_index(chunks):
    global vectorstore, faiss_retriever, bm25_retriever, hybrid_retriever, reranker

    emb = create_embeddings()
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=emb,
    )

    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    vectorstore.save_local("faiss_index")

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 6
    hybrid_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )
    if USE_FAKE_MODELS:
        reranker = FakeCrossEncoder()
    else:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return vectorstore


def enhance_query(question):
    if not GROQ_API_KEY:
        return question

    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=80,
        groq_api_key=GROQ_API_KEY,
    )

    query_expansion_prompt = ChatPromptTemplate.from_template(
        """
You are a search query optimizer for a RAG system about the uploaded document.

Rewrite the user's question so that it is clearer and contains useful keywords
for retrieving relevant document passages.

Return ONLY the improved query.

User Question:
{question}

Improved Query:
"""
    )

    query_expansion_chain = query_expansion_prompt | model | StrOutputParser()

    try:
        improved_query = query_expansion_chain.invoke({"question": question}).strip()
        return improved_query or question
    except Exception:
        return question


def rerank_docs(query, docs):
    if not docs:
        return []

    if reranker is None:
        return docs[:5]

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in ranked[:5]]


def retrieve_docs(question):
    if hybrid_retriever is None:
        return []

    enhanced_query = enhance_query(question)
    docs = hybrid_retriever.invoke(enhanced_query)

    unique_docs = {doc.page_content: doc for doc in docs}.values()
    return rerank_docs(enhanced_query, list(unique_docs))


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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

    prompt = f"""You are a helpful assistant that answers questions about the uploaded document.
IMPORTANT: Try to answer using information from the provided context. If not found, provide related information from the document.

Context from document:
{context}

Conversation history:
{history_text}

User Question:
{question}

STRICT RULES:
1. Answer using ONLY information from the provided context
2. If exact answer is not found, try to provide related information from the document that might help
3. If the question is completely off-topic and has NO relation to the document, respond: "I could not find this information in the document."
4. Do NOT confuse different sections (e.g., authors vs references)
5. Do NOT use phrases like "Based on my knowledge" or "I think"
6. Do NOT mention that you are analyzing or attempting
7. Keep tone confident and natural
8. If possible, cite or reference the relevant part of the document

Answer:"""

    response = model.invoke(prompt)
    return response.content


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global chat_history, sections, uploaded_text

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

    uploaded_text = text

    sections = {
        "authors": extract_authors(text),
        "references": extract_references(text),
        "abstract": extract_abstract(text),
    }

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
    global chat_history, sections, uploaded_text

    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY not set"}), 500

    if vectorstore is None:
        return jsonify({"error": "Upload a PDF first"}), 400

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    intent = detect_intent(question)
    
    if intent in sections:
        answer = sections[intent]
        if not answer and uploaded_text:
            if intent == "authors":
                answer = extract_authors(uploaded_text)
            elif intent == "references":
                answer = extract_references(uploaded_text)
            elif intent == "abstract":
                answer = extract_abstract(uploaded_text)
            if answer:
                sections[intent] = answer

    if intent in sections and sections[intent]:
        answer = sections[intent]
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        chat_history = chat_history[-10:]
        return jsonify({"answer": answer})

    docs = retrieve_docs(question)
    context = format_docs(docs)

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
    app.run(host="0.0.0.0", port=80)
