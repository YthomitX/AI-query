from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import bridge

# Path to your stored policies
storage_path = r"C:\\Users\\TIM BORJA\\AI Project\\medical-ai\\storage\\app\\public\\policies"

# Initialize FAISS index, chunks, and embeddings
index, all_chunks, all_embeddings = bridge.load_pdfs(storage_path)

app = Flask(__name__)
CORS(app)

@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json(silent=True) or {}
        question = str(data.get("question", "")).strip()
        if not question:
            return jsonify({
                "answer": "I couldn’t find relevant information in the uploaded documents.",
                "sources": []
            })

        # Call bridge with retrieval
        answer, sources_used, matched_chunks = bridge.answer_with_mistral(
            index, question, all_chunks, k=10
        )

        # ✅ Debug logs
        print("Question:", question)
        print("Chunks loaded:", len(all_chunks))
        print("Sources available:", {c.get("source") for c in all_chunks})
        for c in matched_chunks:
            print("Chunk source:", c.get("source"), "Score:", c.get("score"))

        if not matched_chunks:
            return jsonify({
                "answer": "I couldn’t find relevant information in the uploaded documents. Please check the official PCG guidelines.",
                "sources": []
            })

        # ✅ Always take top N chunks (multi-document context)
        top_chunks = sorted(matched_chunks, key=lambda c: c.get("score", 0), reverse=True)[:3]

        # ✅ Build combined context from multiple chunks
        combined_answer = answer
        if not combined_answer:
            combined_answer = "\n\n".join(
                [c.get("summary", c.get("text", "")) for c in top_chunks]
            )

        sources = list({c.get("source") for c in top_chunks})

        return jsonify({
            "answer": str(combined_answer),
            "sources": sources
        })
    except Exception as e:
        print(f"AI error: {e}")
        return jsonify({"answer": "Information unavailable", "sources": []})


@app.route("/reindex", methods=["POST"])
def reindex():
    try:
        data = request.get_json(silent=True) or {}
        file_path = data.get("path")
        title = data.get("title")

        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 400

        # Load just the new file
        new_index, new_chunks, new_embeddings = bridge.load_pdfs(file_path)

        global index, all_chunks, all_embeddings

        if index is None:
            index = new_index
            all_chunks = new_chunks
            all_embeddings = new_embeddings
        else:
            # Append new vectors with IDs
            ids = np.array([c["id"] for c in new_chunks], dtype="int64")
            index.add_with_ids(np.array(new_embeddings).astype("float32"), ids)

            all_chunks.extend(new_chunks)
            all_embeddings.extend(new_embeddings)

        return jsonify({"status": "success", "file": title})
    except Exception as e:
        print(f"Reindex error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/reindex_all", methods=["POST"])
def reindex_all():
    try:
        global index, all_chunks, all_embeddings
        storage_path = r"C:\\Users\\TIM BORJA\\AI Project\\medical-ai\\storage\\app\\public\\policies"

        index, all_chunks, all_embeddings = bridge.load_pdfs(storage_path)

        return jsonify({"status": "success", "files_indexed": len(all_chunks)})
    except Exception as e:
        print(f"Bulk reindex error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/delete_policy", methods=["POST"])
def delete_policy():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")

    global all_chunks, all_embeddings, index

    ids_to_remove = [c["id"] for c in all_chunks if c["source"] == filename]

    if ids_to_remove and index is not None:
        index.remove_ids(np.array(ids_to_remove, dtype="int64"))

        filtered_pairs = [(c, e) for c, e in zip(all_chunks, all_embeddings) if c["source"] != filename]
        all_chunks = [c for c, _ in filtered_pairs]
        all_embeddings = [e for _, e in filtered_pairs]
    else:
        index, all_chunks, all_embeddings = None, [], []

    return jsonify({"status": "deleted", "filename": filename})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
