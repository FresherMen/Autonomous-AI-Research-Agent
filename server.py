import os
from dotenv import load_dotenv
load_dotenv()   # reads .env and puts the key into os.environ before imports

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from retriever import RAGRetriever
from agent import ResearchAgent
from report_generator import ReportGenerator
from indexer import DocumentIndexer
from tools import ToolExecutor

app = FastAPI(title="Research Assistant API")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Shared instances (loaded once at startup to optimize memory & prevent locks)
indexer = DocumentIndexer()
rag     = RAGRetriever(indexer=indexer)
executor = ToolExecutor(rag=rag)
agent   = ResearchAgent(executor=executor)
reports = ReportGenerator(indexer=indexer)

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
        "tool_calls": getattr(agent, "last_tool_calls", []),
        "sources":    getattr(agent, "last_sources", [])
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
    indexed = []
    for f in files:
        filename = os.path.basename(f.filename or "uploaded_file")
        path = os.path.join("./docs", filename)
        with open(path, "wb") as out:
            out.write(await f.read())
        try:
            indexer.index_file(path)
            indexed.append(filename)
        except Exception as e:
            print(f"Error indexing {filename}: {e}")
    return {"indexed": indexed}

@app.get("/docs")
async def list_docs():
    try:
        results = indexer.store.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []
        docs = sorted(set(
            m.get("source") for m in metadatas 
            if m and isinstance(m, dict) and "source" in m
        ))
        return {"docs": docs}
    except Exception as e:
        print(f"Error listing docs: {e}")
        return {"docs": []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)