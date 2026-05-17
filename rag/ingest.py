import json
import os
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
POLICIES_PATH = os.path.join(DATA_DIR, "policies.json")
FAISS_PATH = os.path.join(os.path.dirname(__file__), "store.faiss")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "chunks.pkl")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_policies():
    with open(POLICIES_PATH, "r") as f:
        return json.load(f)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest_text_files(policies):
    chunks = []
    docs_dir = os.path.join(DATA_DIR, "docs")
    for fname in os.listdir(docs_dir):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(docs_dir, fname)
        with open(fpath, "r") as f:
            text = f.read()
        policy = policies.get(fname, {})
        allowed_roles = policy.get("allowed_roles", [])
        for chunk_text_piece in chunk_text(text):
            chunks.append({
                "text": chunk_text_piece,
                "source": fname,
                "allowed_roles": allowed_roles,
            })
    return chunks


def ingest_csv_files(policies):
    chunks = []
    csvs_dir = os.path.join(DATA_DIR, "csvs")
    for fname in os.listdir(csvs_dir):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(csvs_dir, fname)
        df = pd.read_csv(fpath)
        policy = policies.get(fname, {})
        allowed_roles = policy.get("allowed_roles", [])
        for _, row in df.iterrows():
            parts = " | ".join(f"{col}={row[col]}" for col in df.columns)
            text = f"[{fname}] {parts}"
            chunks.append({
                "text": text,
                "source": fname,
                "allowed_roles": allowed_roles,
            })
    return chunks


def ingest_json_logs(policies):
    chunks = []
    log_path = os.path.join(DATA_DIR, "logs", "audit.json")
    if not os.path.exists(log_path):
        return chunks
    with open(log_path, "r") as f:
        entries = json.load(f)
    policy = policies.get("audit.json", {})
    allowed_roles = policy.get("allowed_roles", [])
    for entry in entries:
        text = f"[audit.json] {json.dumps(entry)}"
        chunks.append({
            "text": text,
            "source": "audit.json",
            "allowed_roles": allowed_roles,
        })
    return chunks


def main():
    print("Loading policies...")
    policies = load_policies()

    print("Ingesting documents...")
    all_chunks = []
    all_chunks.extend(ingest_text_files(policies))
    all_chunks.extend(ingest_csv_files(policies))
    all_chunks.extend(ingest_json_logs(policies))

    print(f"Total chunks: {len(all_chunks)}")

    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Saved {FAISS_PATH} ({index.ntotal} vectors)")
    print(f"Saved {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
