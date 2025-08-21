from __future__ import annotations

import os
from flask import Flask, request, jsonify, send_from_directory, render_template

import config
from embedding_utils import Embedder, VectorStore

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="data/processed",  # serve processed images under /static/images/
    static_url_path="/static",
)

# Load vector store
vector_store = VectorStore(dim=config.EMBED_DIM)
vector_store.load()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    q_vec = Embedder.embed_text([query])
    results = vector_store.search(q_vec, top_k=config.TOP_K)

    # Transform image paths to URLs
    for r in results:
        if r["type"] == "image":
            # build URL relative to /static
            filename = os.path.basename(r["image_path"])
            r["url"] = f"/static/images/{filename}"
            del r["image_path"]
        if "content" in r:
            # Optionally truncate long text for preview
            r["content"] = r["content"].strip()
    return jsonify({"results": results})


# Provide dedicated route to serve images if needed (not used when static serving)
@app.route("/image/<path:filename>")
def image(filename):
    return send_from_directory(config.IMAGE_DIR, filename)


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)