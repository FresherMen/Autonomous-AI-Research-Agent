import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from retriever import RAGRetriever
from agent import ResearchAgent
from report_generator import ReportGenerator
from indexer import DocumentIndexer
from dotenv import load_dotenv
load_dotenv()   # reads .env and puts the key into os.environ
app = FastAPI(title="Research Assistant API")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Shared instances (loaded once at startup)
rag     = RAGRetriever()
agent   = ResearchAgent()
reports = ReportGenerator()
indexer = DocumentIndexer()

class Question(BaseModel):
    question: str

class ReportRequest(BaseModel):
    type:    str
    topic:   str
    topic_b: str = ""

@app.post("/ask")
async def ask(q: Question):
    result = rag.ask(q.question)
    return {
        "answer":  result["answer"],
        "sources": result["chunks"],
        "tokens":  result["tokens"]
    }

@app.post("/agent")
async def ask_agent(q: Question):
    answer = agent.run(q.question)
    return {
        "answer":     answer,
        "tool_calls": [],   # extend agent.py to capture these if you want
        "sources":    []
    }

@app.post("/report")
async def generate_report(r: ReportRequest):
    if r.type == "bullet_summary":
        data     = reports.bullet_summary(r.topic)
        rendered = reports.render_bullet_summary(data)
    elif r.type == "research_brief":
        data     = reports.research_brief(r.topic)
        rendered = reports.render_research_brief(data)
    elif r.type == "comparison":
        data     = reports.comparison(r.topic, r.topic_b)
        rendered = reports.render_comparison(data)
    else:
        rendered = "Unknown report type"
    return {"rendered": rendered}

@app.post("/index")
async def index_files(files: list[UploadFile] = File(...)):
    os.makedirs("./docs", exist_ok=True)
    for f in files:
        path = f"./docs/{f.filename}"
        with open(path, "wb") as out:
            out.write(await f.read())
        indexer.index_file(path)
    return {"indexed": [f.filename for f in files]}

@app.get("/docs")
async def list_docs():
    if not os.path.exists("./chroma_db"):
        return {"docs": []}
    from vector_store import VectorStore
    store = VectorStore()
    # Get unique source names from the collection
    results = store.collection.get(include=["metadatas"])
    docs = sorted(set(m["source"] for m in results["metadatas"]))
    return {"docs": docs}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)