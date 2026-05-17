import datetime
import json
import os
import pickle

from dotenv import load_dotenv
load_dotenv()

import faiss
import numpy as np
from fastapi import FastAPI
from groq import Groq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

APP_DIR = os.path.dirname(__file__)
FAISS_PATH = os.path.join(APP_DIR, "store.faiss")
CHUNKS_PATH = os.path.join(APP_DIR, "chunks.pkl")
AUDIT_LOG_PATH = os.path.join(APP_DIR, "audit.log")

PROMPT_TEMPLATE = """You are an enterprise assistant. Answer ONLY from the context below.
Cite every fact as [source]. If context is insufficient, say "I don't have enough information."
Do NOT use outside knowledge.

Context:
{ctx}

Question: {q}
Answer:"""

app = FastAPI(title="RAG Intelligence")

# Load resources at startup
index = faiss.read_index(FAISS_PATH)
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
groq_client = Groq()


class AskRequest(BaseModel):
    question: str
    role: str


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    denied_sources: int
    chunks_used: int


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # Embed query
    q_embedding = model.encode([req.question])
    q_embedding = np.array(q_embedding, dtype="float32")

    # FAISS search — over-fetch top 15
    distances, indices = index.search(q_embedding, 15)

    # RBAC filter
    allowed = []
    denied = 0
    for idx in indices[0]:
        if idx < 0:
            continue
        chunk = chunks[idx]
        if req.role in chunk["allowed_roles"]:
            if len(allowed) < 5:
                allowed.append(chunk)
        else:
            denied += 1

    # If no allowed chunks, return refusal
    if not allowed:
        answer = "No accessible information for your role."
        citations = []
        _log_audit(req.role, req.question, citations, denied)
        return AskResponse(
            answer=answer,
            citations=citations,
            denied_sources=denied,
            chunks_used=0,
        )

    # Build context
    ctx = "\n\n".join(f"[{c['source']}] {c['text']}" for c in allowed)

    # Call Groq LLM
    prompt = PROMPT_TEMPLATE.format(ctx=ctx, q=req.question)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    answer = response.choices[0].message.content

    # Unique citations
    citations = list(dict.fromkeys(c["source"] for c in allowed))

    _log_audit(req.role, req.question, citations, denied)

    return AskResponse(
        answer=answer,
        citations=citations,
        denied_sources=denied,
        chunks_used=len(allowed),
    )


def _log_audit(role: str, question: str, sources: list[str], denied: int):
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    entry = f"{ts} | {role} | {question} | {json.dumps(sources)} | denied={denied}\n"
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(entry)
