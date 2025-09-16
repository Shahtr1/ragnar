# src/ingest.py
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import sys

# Paths 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INDEX_PATH = PROJECT_ROOT / "ragnar_index.faiss"
DATA_JSON = PROJECT_ROOT / "ragnar_data.json"

MODEL_NAME = "all-MiniLM-L6-v2"

def read_and_chunk_txt_files(data_dir: Path):
    texts = []
    metadatas = []
    txt_files = sorted(list(data_dir.glob("*.txt")))
    if not txt_files:
        print(f"[ERROR] No .txt files found in {data_dir}. Create at least one .txt (eg: data/test1.txt).")
        return texts, metadatas

    for f in txt_files:
        raw = f.read_text(encoding="utf-8", errors="ignore")
        # naive paragraph split; you can replace with better chunker later
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        if not paragraphs:
            # fallback: split by lines
            paragraphs = [line.strip() for line in raw.splitlines() if line.strip()]
        for i, p in enumerate(paragraphs):
            texts.append(p)
            metadatas.append({"source": f.name, "chunk": i})
    return texts, metadatas

def main():
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"Reading data from {DATA_DIR}")
    texts, metadatas = read_and_chunk_txt_files(DATA_DIR)

    print(f"Found {len(texts)} chunks from {len(set(m['source'] for m in metadatas))} files.")
    if len(texts) == 0:
        sys.exit(1)

    print("Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    # encode -> force numpy 2D array
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    emb_np = np.array(embeddings).astype("float32")

    # Ensure emb_np is 2D
    if emb_np.ndim == 1:
        # only one vector returned -> make it (1, dim)
        emb_np = emb_np.reshape(1, -1)
    if emb_np.ndim != 2:
        print("[ERROR] embeddings array has unexpected shape:", emb_np.shape)
        sys.exit(1)

    n, d = emb_np.shape
    print(f"Embeddings shape: n={n}, dim={d}")

    # Build FAISS index (flat L2 for demo)
    index = faiss.IndexFlatL2(d)
    index.add(emb_np)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saved FAISS index with {index.ntotal} vectors to {INDEX_PATH}")

    # Save texts and metadata for provenance lookup
    with open(DATA_JSON, "w", encoding="utf-8") as fh:
        json.dump({"texts": texts, "metadatas": metadatas}, fh, ensure_ascii=False, indent=2)
    print(f"Saved chunk data to {DATA_JSON}")

if __name__ == "__main__":
    main()
