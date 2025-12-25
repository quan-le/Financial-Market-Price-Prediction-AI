from flask import Flask, request, jsonify
import numpy as np
from pathlib import Path
from fed_speech_updater import SpeechDecayBuilder

app = Flask(__name__)
OUT_DIR = Path("../data/processed").resolve()
DAILY_DIR = OUT_DIR / "daily_embeddings"

builder = SpeechDecayBuilder(load_existing=True)

@app.route('/add', methods=['POST'])
def add_doc():
    data = request.json
    builder.add_single_speech(
        data['text'],
        data['release_date'],
        data['document_kind']
    )
    return jsonify({"status": "ok"})


@app.route('/embedding/<date>')
def get_embedding(date):
    file = DAILY_DIR / f"{date}_embeddings.npz"
    if not file.exists():
        return jsonify({"error": "Not found"}), 404
    data = np.load(file)
    return jsonify({"embedding": data['embedding'].tolist()})


if __name__ == '__main__':
    app.run(debug=True, port=8000)
