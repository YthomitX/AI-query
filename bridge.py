import os
import pytesseract
import faiss
import numpy as np
import requests
from PyPDF2 import PdfReader

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def chunk_text(text, size=800, overlap=100):
    """Split text into overlapping chunks for better coverage of long PDFs."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += size - overlap
    return chunks

def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        payload = {"model": "nomic-embed-text", "prompt": chunk}
        try:
            response = requests.post("http://localhost:11434/api/embeddings", json=payload)
            data = response.json()

            if "embedding" in data:
                embeddings.append(data["embedding"])
            elif "data" in data and len(data["data"]) > 0 and "embedding" in data["data"][0]:
                embeddings.append(data["data"][0]["embedding"])
            else:
                print("⚠️ No embedding returned for chunk:", chunk[:50])
        except Exception as e:
            print(f"Embedding error: {e}")
    return embeddings

def summarize_chunk(chunk_text):
    """Generate a concise summary for a chunk using Mistral."""
    try:
        payload = {
            "model": "mistral",
            "prompt": f"Summarize this text in 2-3 sentences, focusing on key points:\n\n{chunk_text}",
            "stream": False
        }
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        data = response.json()
        return data.get("response", chunk_text[:200])  # fallback: first 200 chars
    except Exception as e:
        print(f"Summarization error: {e}")
        return chunk_text[:200]

def store_in_faiss(embeddings, chunks):
    vectors = np.array(embeddings).astype("float32")
    dimension = vectors.shape[1]

    base_index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(base_index)

    ids = []
    for i, c in enumerate(chunks):
        unique_id = abs(hash(f"{c['source']}_{i}")) % (10**12)
        ids.append(unique_id)
        c["id"] = unique_id

    ids = np.array(ids, dtype="int64")
    index.add_with_ids(vectors, ids)

    print(f"✅ FAISS index populated with {index.ntotal} vectors")
    return index, chunks

def answer_with_mistral(index, query, chunks, k=10, top_n=3):
    SYSTEM_PROMPT = """
    You are a helpful, knowledgeable AI assistant designed to support users in medical, technical, and everyday queries.
    Your personality is friendly, professional, and clear.
    You always provide accurate, complete, and structured answers.

    You are given excerpts from uploaded policy documents. Use ONLY these excerpts to answer the user’s question.
    Always identify which document the information came from by name.
    If the context does not contain relevant details, say honestly that the answer is not available in the provided documents.
    """

    try:
        query = query.lower().strip()

        # Embed query
        payload = {"model": "nomic-embed-text", "prompt": query}
        response = requests.post("http://localhost:11434/api/embeddings", json=payload)
        data = response.json()

        if "embedding" in data:
            query_vector = np.array(data["embedding"]).astype("float32").reshape(1, -1)
        elif "data" in data and len(data["data"]) > 0 and "embedding" in data["data"][0]:
            query_vector = np.array(data["data"][0]["embedding"]).astype("float32").reshape(1, -1)
        else:
            return "Information unavailable", [], []

        # Search FAISS
        distances, indices = index.search(query_vector, k=k)
        if indices is None or len(indices[0]) == 0:
            return "Information unavailable", [], []

        id_to_chunk = {c["id"]: c for c in chunks}
        retrieved = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in id_to_chunk:
                chunk = id_to_chunk[idx]
                similarity = 1 / (1 + float(dist))  # convert distance to similarity
                chunk["score"] = similarity
                retrieved.append(chunk)

        # Debug
        for c in retrieved:
            print(f"Retrieved chunk from {c['source']} (page {c.get('page','?')}) "
                  f"score={c['score']:.3f}\nText preview: {c['text'][:150]}...\n")

        # Select top N chunks
        top_chunks = sorted(retrieved, key=lambda c: c["score"], reverse=True)[:top_n]

        if not top_chunks:
            return "I couldn’t find relevant information in the uploaded documents.", [], []

        # Build combined context
        context = "\n\n".join(
            [f"From {c['source']} (page {c.get('page','?')}):\n{c.get('summary', c['text'])}"
             for c in top_chunks]
        )

        full_prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

User Question: {query}

Answer based only on the context and your role above.
"""

        payload = {"model": "mistral", "prompt": full_prompt, "stream": False}
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        data = response.json()
        sources = list({c["source"] for c in top_chunks})
        return data.get("response", "Information unavailable"), sources, top_chunks

    except Exception as e:
        print(f"Answer error: {e}")
        return "Information unavailable", [], []

def load_pdfs(path):
    all_chunks = []
    all_embeddings = []

    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".pdf")]
    else:
        files = [path]

    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                for chunk in chunk_text(text):
                    emb_list = embed_chunks([chunk])
                    if emb_list:
                        summary = summarize_chunk(chunk)
                        all_chunks.append({
                            "text": chunk,
                            "summary": summary,
                            "source": file_name,
                            "page": page_num
                        })
                        all_embeddings.append(emb_list[0])
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    if all_embeddings:
        index, chunks = store_in_faiss(all_embeddings, all_chunks)
        return index, chunks, all_embeddings
    else:
        print("⚠️ No embeddings generated. Check your PDFs.")
        return None, [], []
