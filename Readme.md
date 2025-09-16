## Libraries

- Extract text from files: `pdfminer.six`, `PyPDF2`
- Chunk & embed text: `sentence-transformers`, `numpy`, tik`token (for token counting)
- Store & search vectors: `faiss-cpu`
- Connect to LLMs / tooling: `transformers` (and later other local LLM wrappers)
- Orchestration / utilities: `langchain` (optional glue/helpers)

1. `faiss-cpu`

In AI and Machine Learning, we often represent things (like text, images, or users) as embeddings → which are just vectors (lists of numbers).
When you have millions of vectors, you need a way to search quickly for "which ones are most similar."

### How FAISS works (intuition)

If you have a database of 1 million vectors (say, 512 dimensions each), and you want to find which vectors are closest to a query vector, a simple brute-force comparison would be very slow.

FAISS provides:

1.  Efficient data structures (indexes) to store vectors.
2.  Fast search algorithms (like Approximate Nearest Neighbor search).

This makes similarity search much faster, even at massive scale.

2. `sentence-transformers`

3. `transformers`

4. `langchain`

5. `pdfminer.six` and
6. `PyPDF2`

- Both extract text from PDF files. We need them to ingest PDFs into the RAG pipeline.
- `pdfminer.six` is geared toward extracting _text_ and _layout_ (often better for text-heavy PDFs).
- `PyPDF2` is simpler for splitting/merging PDFs and extracting pages/text; sometimes less accurate on complex layouts.

7. `tiktoken`

8) `numpy`

- Fundamental numerical array library. We use `numpy` arrays to store embeddings before adding them to FAISS or saving them.
- FAISS expects `numpy.float32` arrays.
