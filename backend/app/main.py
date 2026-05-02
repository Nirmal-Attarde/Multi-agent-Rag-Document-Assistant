from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from .orchestrator import Orchestrator

load_dotenv()

app = FastAPI(title="Multi-Agent RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing orchestrator...")
orchestrator = Orchestrator()
print("Orchestrator ready.")


class ChatRequest(BaseModel):
    message: str


class Source(BaseModel):
    source: str
    score: float
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    trace: dict


@app.get("/")
def root():
    return {"status": "ok", "message": "Multi-Agent RAG API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = orchestrator.run(request.message)
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        trace=result["trace"],
    )