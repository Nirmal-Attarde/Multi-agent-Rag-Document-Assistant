# import os
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from groq import Groq

# load_dotenv()

# app = FastAPI(title="Multi-Agent RAG API")

# # Allow the Next.js frontend (running on a different port) to call this API
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# class ChatRequest(BaseModel):
#     message: str


# class ChatResponse(BaseModel):
#     reply: str


# @app.get("/")
# def root():
#     return {"status": "ok", "message": "Multi-Agent RAG API is running"}


# @app.post("/chat", response_model=ChatResponse)
# def chat(request: ChatRequest):
#     response = groq_client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": request.message},
#         ],
#     )
#     return ChatResponse(reply=response.choices[0].message.content)


# New

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from .rag_agent import RAGAgent

load_dotenv()

app = FastAPI(title="Multi-Agent RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate once at startup. The embedding model loads here (~5 seconds).
print("Initializing RAG agent...")
rag_agent = RAGAgent()
print("RAG agent ready.")


class ChatRequest(BaseModel):
    message: str
    top_k: int = 5


class Source(BaseModel):
    source: str
    score: float
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/")
def root():
    return {"status": "ok", "message": "Multi-Agent RAG API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = rag_agent.answer(request.message, top_k=request.top_k)
    return ChatResponse(answer=result["answer"], sources=result["sources"])