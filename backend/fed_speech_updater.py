import os
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from datetime import timezone

OUT_DIR = Path("../data/processed").resolve()
DAILY_DIR = OUT_DIR / "daily_embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DAILY_DIR.mkdir(parents=True, exist_ok=True)

DOC_PRIORS = {
    "fomc": {"weight": 1.00, "half_life": 10},
    "minutes": {"weight": 0.95, "half_life": 12},
    "press_conferences": {"weight": 0.90, "half_life": 7},
    "transcripts": {"weight": 0.85, "half_life": 20},
    "projections": {"weight": 1.00, "half_life": 30},
    "speeches": {"weight": 0.60, "half_life": 5},
    "beige_book": {"weight": 0.45, "half_life": 4},
    "redbooks": {"weight": 0.35, "half_life": 3},
    "teal_book": {"weight": 0.30, "half_life": 5},
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    return text.strip().lower()


def half_life_from_text(text: str, base: float) -> float:
    keywords = [
        "uncertainty", "outlook", "forecast", "risks",
        "expected", "projected", "anticipate"
    ]
    boost = sum(1 for k in keywords if k in text)
    return base * (1 + 0.15 * boost)


def decay_weight(days: int, half_life: float) -> float:
    return math.exp(-math.log(2) * days / half_life)


class SpeechDecayBuilder:
    def __init__(self, load_existing=False):
        if load_existing:
            self.df = pd.read_csv(OUT_DIR / "speech_metadata.csv")
            self.df["release_date"] = pd.to_datetime(self.df["release_date"])
        else:
            raise ValueError("load_existing must be True for incremental updates")

        self.embedder = SentenceTransformer("yiyanghkust/finbert-tone")
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()
        self.backend = "sentence-transformers"

    def build_daily_embeddings(self, embeddings: np.ndarray, start_from=None):
        if start_from:
            start = pd.to_datetime(start_from).date()
        else:
            start = self.df["release_date"].min().date()

        end = datetime.now(timezone.utc).date()

        print(f"Aggregating daily vectors from {start} to {end}...")

        for day in pd.date_range(start, end):
            vec = np.zeros(embeddings.shape[1])
            mask = self.df["release_date"].dt.date <= day.date()
            relevant_docs = self.df[mask]

            for _, row in relevant_docs.iterrows():
                days_diff = (day.date() - row["release_date"].date()).days
                prior = DOC_PRIORS.get(row["document_kind"], {"weight": 0.3, "half_life": 5})
                hl = half_life_from_text(row["clean_text"], prior["half_life"])
                w = prior["weight"] * decay_weight(days_diff, hl)
                vec += w * embeddings[int(row["doc_index"])]

            if np.linalg.norm(vec) > 0:
                vec /= np.linalg.norm(vec)

            out = DAILY_DIR / f"{day.date()}_embeddings.npz"
            np.savez_compressed(
                out,
                embedding=vec,
                date=str(day.date()),
                backend=self.backend
            )

    def add_single_speech(self, text, release_date, doc_kind):
        clean = clean_text(text)
        new_vec = self.embedder.encode([clean])

        current_embeddings = np.load(OUT_DIR / "embeddings.npy")
        updated_embeddings = np.vstack([current_embeddings, new_vec])
        np.save(OUT_DIR / "embeddings.npy", updated_embeddings)

        new_row = {
            "release_date": pd.to_datetime(release_date),
            "document_kind": doc_kind,
            "clean_text": clean,
            "doc_index": len(updated_embeddings) - 1
        }

        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.df.to_csv(OUT_DIR / "speech_metadata.csv", index=False)

        self.build_daily_embeddings(updated_embeddings, start_from=release_date)


def update_fed_data(new_text, new_date, doc_type):
    updater = SpeechDecayBuilder(load_existing=True)
    updater.add_single_speech(
        text=new_text,
        release_date=new_date,
        doc_kind=doc_type
    )
    print("Fed embeddings and daily decay vectors updated successfully.")


if __name__ == "__main__":
    sample_text = "The committee expects inflation to remain near the 2 percent target..."
    update_fed_data(sample_text, "2025-12-21", "speeches")
