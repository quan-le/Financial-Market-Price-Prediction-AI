from flask import Flask, request, jsonify
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd
import traceback
import joblib

load_dotenv()

from fed_speech_updater import SpeechDecayBuilder
from model_inference import predict_from_arrays
from data_fetcher import AlphaVantagePriceFetcher, EconIndicatorsFetcher
from feature_engineering import engineer_features, prepare_model_input

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = PROJECT_ROOT / "data" / "processed" / "daily_embeddings"
SCALER_DIR = PROJECT_ROOT / "src" / "Model" / "scalers"

print(f"[INFO] Fed embeddings directory: {DAILY_DIR}")
print(f"[INFO] Directory exists: {DAILY_DIR.exists()}")

builder = SpeechDecayBuilder(load_existing=True)
price_fetcher = AlphaVantagePriceFetcher()
econ_fetcher = EconIndicatorsFetcher()

feature_scaler = None
target_scaler = None
try:
    feature_scaler = joblib.load(SCALER_DIR / "feature_scaler.pkl")
    target_scaler = joblib.load(SCALER_DIR / "target_scaler.pkl")
    print(f"[INFO] Scalers loaded successfully")
except Exception as e:
    print(f"[WARNING] Could not load scalers: {e}")


@app.route("/add", methods=["POST"])
def add_doc():
    data = request.json
    builder.add_single_speech(
        data["text"],
        data["release_date"],
        data["document_kind"]
    )
    return jsonify({"status": "ok"})


@app.route("/embedding/<date>")
def get_embedding(date):
    file = DAILY_DIR / f"{date}_embeddings.npz"
    if not file.exists():
        return jsonify({"error": "Not found"}), 404
    data = np.load(file)
    return jsonify({"embedding": data["embedding"].tolist()})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("[INFO] Fetching price data...")
        price_df = price_fetcher.fetch("BTC-USD", asset_type="crypto")
        if price_df is None or price_df.empty:
            return jsonify({"error": "Failed to fetch price data"}), 500
        print(f"[INFO] Price data shape: {price_df.shape}")

        print("[INFO] Fetching economic data...")
        econ_df_raw = econ_fetcher.fetch_all()
        econ_df = econ_fetcher.generate_daily_econ()
        if econ_df is None or econ_df.empty:
            print("[WARNING] No economic data - using empty")
            econ_df = pd.DataFrame()
        else:
            print(f"[INFO] Economic data shape: {econ_df.shape}")

        print("[INFO] Loading Fed embeddings...")
        embedding_files = sorted(DAILY_DIR.glob("*_embeddings.npz"))
        if not embedding_files:
            return jsonify({"error": "No Fed embeddings found"}), 500
        print(f"[INFO] Found {len(embedding_files)} embedding files")

        fed_data = []
        for file in embedding_files[-200:]:
            date_str = file.stem.replace("_embeddings", "")
            emb = np.load(file)["embedding"]
            fed_row = {"date": pd.to_datetime(date_str)}
            for i, val in enumerate(emb):
                fed_row[f"fed_emb_{i}"] = float(val)
            fed_data.append(fed_row)
        fed_df = pd.DataFrame(fed_data)
        print(f"[INFO] Fed embeddings shape: {fed_df.shape}")

        print("[INFO] Engineering features...")
        full_df = engineer_features(price_df, econ_df, fed_df)
        print(f"[INFO] Full features shape: {full_df.shape}")

        print("[INFO] Preparing model input...")
        obs, known, static = prepare_model_input(full_df, lookback=89)

        print("[INFO] Cleaning input data...")
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        known = np.nan_to_num(known, nan=0.0, posinf=1e6, neginf=-1e6)

        if feature_scaler is not None:
            print("[INFO] Scaling features...")
            combined = np.concatenate([obs, known], axis=1)
            combined_scaled = feature_scaler.transform(combined.reshape(-1, combined.shape[-1])).reshape(combined.shape)
            obs_scaled = combined_scaled[:, :obs.shape[1]]
            known_scaled = combined_scaled[:, obs.shape[1]:]
        else:
            print("[WARNING] No feature scaler available - using unscaled features")
            obs_scaled = obs
            known_scaled = known

        print("[INFO] Making prediction...")
        prediction_scaled = predict_from_arrays(obs_scaled, known_scaled, static)

        if target_scaler is not None and not np.isnan(prediction_scaled):
            prediction = float(target_scaler.inverse_transform([[prediction_scaled]])[0, 0])
            print(f"[INFO] Scaled prediction: {prediction_scaled:.4f}, Unscaled: {prediction:.4f}")
        else:
            prediction = prediction_scaled
            print(f"[WARNING] No target scaler - using scaled prediction: {prediction}")

        return jsonify({
            "prediction": float(prediction),
            "timestamp": str(full_df['date'].iloc[-1]) if 'date' in full_df.columns and not full_df.empty and pd.notna(
                full_df['date'].iloc[-1]) else str(pd.Timestamp.now()),
            "features": {
                "n_obs": list(obs.shape),
                "n_known": list(known.shape),
                "n_static": len(static)
            },
            "current_price": float(full_df['close'].iloc[-1]) if 'close' in full_df.columns else None,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }), 500


@app.route("/debug")
def debug():
    return jsonify({
        "fed_dir": str(DAILY_DIR),
        "fed_files": [f.name for f in DAILY_DIR.glob("*_embeddings.npz")],
        "daily_dir_exists": DAILY_DIR.exists(),
        "scalers_loaded": feature_scaler is not None and target_scaler is not None
    })


if __name__ == "__main__":
    app.run(debug=True, port=8000)
