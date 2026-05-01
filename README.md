<p align="center">
	<img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" />
	<img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C" />
	<img src="https://img.shields.io/badge/FAISS-Vector%20Store-009688" />
	<img src="https://img.shields.io/badge/HuggingFace-Embeddings-FFB000?logo=huggingface" />
	<img src="https://img.shields.io/badge/Groq-LLM-000000" />
</p>

# 📚 RAG Notebook Studio
### *(Document Ingestion • Vector Search • Conversational RAG • Streaming Responses)*

**Short description:** Notebook-first RAG system using LangChain, FAISS, HuggingFace embeddings, and Groq LLMs.

A hands-on Retrieval-Augmented Generation (RAG) project built in Jupyter Notebook format.  
It loads local documents, chunks and embeds them, stores vectors in FAISS, and answers queries using Groq-hosted LLMs with both standard and conversational flows.

This project demonstrates practical GenAI engineering patterns: document pipelines, semantic retrieval, prompt-driven answering, and conversational context handling.

---

## 🔥 Features

- 📥 **Document Ingestion**: Load `.docx` files from a local data directory
- ✂️ **Text Chunking**: Recursive chunk splitting with overlap for better retrieval
- 🧠 **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via LangChain HuggingFace integration
- 🗂️ **FAISS Vector Store**: Fast similarity search with persistent local index
- 🔎 **Similarity Search**: Query top-k relevant chunks with optional relevance scores
- 🤖 **LLM Integration**: Groq chat model (`llama-3.1-8b-instant`) for answer generation
- 💬 **Conversational RAG**: Maintains chat history using LangChain message objects
- 🌊 **Streaming Output**: Token streaming for incremental response display
- 🧪 **Notebook-First Workflow**: End-to-end experimentation and debugging in `rag_demo.ipynb`

---

## 🎓 How It Works

1. **Load documents** from `RAG_System/data/`
2. **Split into chunks** using `RecursiveCharacterTextSplitter`
3. **Generate embeddings** for each chunk
4. **Build FAISS index** and save locally
5. **Retrieve top-k chunks** for each query
6. **Prompt LLM with retrieved context**
7. **Return final answer** (simple, streaming, or conversational)

---

## 🧰 Tech Stack

- **Python 3.13+**
- **LangChain** (core pipeline + LCEL)
- **FAISS** (`faiss-cpu`) for vector similarity search
- **HuggingFace Embeddings** (`langchain-huggingface` + `sentence-transformers`)
- **Groq API** (`langchain-groq`) for LLM inference
- **Jupyter Notebook** for development workflow
- **python-dotenv** for secure environment loading

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

- Python 3.13+
- Groq API key

### 2️⃣ Clone Repository

```bash
git clone https://github.com/your-username/project-1.git
cd project-1
```

### 3️⃣ Install Dependencies

#### Option A (recommended): `uv`
```bash
uv sync
```

#### Option B: `pip`
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

### 4️⃣ Configure Environment

Create `.env` from template:

```bash
copy .env.example .env
```

Set your key:

```env
GROQ_API_KEY=your_real_key_here
```

### 5️⃣ Run Notebook

Open:
- `RAG_System/rag_demo.ipynb`

Run cells top-to-bottom to execute the complete pipeline.

---

## 📁 Project Structure

```text
Project_1/
├── main.py
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── .env.example
├── .gitignore
└── RAG_System/
		├── rag_demo.ipynb
		├── data/                # local documents (ignored except .gitkeep)
		│   └── .gitkeep
		└── faiss_index/         # generated locally (ignored)
```

---

## 🧪 RAG Modes in Notebook

- **Simple RAG Chain**: One-shot context + answer
- **Streaming RAG Chain**: Incremental token generation
- **Conversational RAG Chain**: Multi-turn Q&A with `chat_history`

---

## 🛠️ Troubleshooting

### Missing API Key
- Ensure `.env` exists in project root
- Confirm `GROQ_API_KEY` is set correctly

### No Documents Loaded
- Place `.docx` files inside `RAG_System/data/`
- Verify notebook is run from project root context

### FAISS Load Issues
- Rebuild index cells if embedding model changed
- Ensure `allow_dangerous_deserialization=True` is used only for trusted local files

---

## ⚠️ GitHub Notes

- `.env` and local secret files are ignored
- `RAG_System/faiss_index/` is ignored (generated artifact)
- `RAG_System/data/*` is ignored, except `RAG_System/data/.gitkeep`

---

## 🔮 Future Improvements

- Add evaluation metrics (precision@k, response faithfulness)
- Add hybrid retrieval (BM25 + dense vectors)
- Add reranking stage for better answer quality
- Convert notebook flow into modular Python package/API
- Add automated tests for ingestion and retrieval steps

---

## 🐳 Docker & EC2 Deployment

This project includes a production-ready Dockerfile. The container runs with `gunicorn` and exposes port `80`.

Build the Docker image locally:

```bash
docker build -t rag-notebook-studio:latest .
```

Run the container (recommended: mount a host volume for the FAISS index so it persists between runs):

```bash
# create a local folder for the faiss index
mkdir -p ./faiss_index

docker run -d \
	-p 80:80 \
	-e GROQ_API_KEY=your_key_here \
	-v $(pwd)/faiss_index:/app/faiss_index \
	--name rag-studio \
	rag-notebook-studio:latest
```

Notes for AWS EC2 deployment:
1. Provision an EC2 instance (Ubuntu 22.04 LTS recommended) and install Docker.
2. Copy the image to the EC2 host (either build there or push to a registry and pull).
3. On the EC2 host, run the container as above and ensure port 80 is open in the security group.

Example quick commands (on EC2):

```bash
# Pull the image from Docker Hub (if you pushed it) or build on the instance
docker pull your-dockerhub-username/rag-notebook-studio:latest

docker run -d -p 80:80 \
	-e GROQ_API_KEY="$GROQ_API_KEY" \
	-v /home/ubuntu/faiss_index:/app/faiss_index \
	--restart unless-stopped \
	--name rag-studio \
	your-dockerhub-username/rag-notebook-studio:latest
```

Security & production tips:
- Do not bake secrets into images. Always pass `GROQ_API_KEY` via environment variables or a secrets manager.
- Mount persistent storage for `faiss_index/` so embeddings survive container restarts.
- Use a process manager or container orchestrator (systemd, Docker Compose, or ECS) to auto-restart and manage the service.
- If resource constrained, reduce `gunicorn` workers (default in Dockerfile is 4). For CPU-heavy embedding steps, consider offloading embedding generation or using smaller instances.


## 👨‍💻 Author

Built by **Shounak** for AI coursework and practical RAG experimentation.

---

## ⭐ If this project helped you, consider starring the repository!

