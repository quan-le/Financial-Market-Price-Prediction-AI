import os
from pathlib import Path
import numpy as np
import torch
from mini_tft import MiniTFT

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None


def _get_required_int(name: str) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        raise ValueError(f"Missing required env var: {name}")
    return int(v)


def _get_model_path() -> Path:
    v = os.getenv("MINI_TFT_MODEL_PATH")
    if v is not None and str(v).strip() != "":
        return Path(v).expanduser().resolve()
    return (Path(__file__).resolve().parent / "src/Model/mini_tft_model.pth").resolve()


def get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_path = _get_model_path()
    n_obs = _get_required_int("MINI_TFT_NOBS")
    n_known = _get_required_int("MINI_TFT_NKNOWN")
    d_static = int(os.getenv("MINI_TFT_DSTATIC", "2"))
    d_model = int(os.getenv("MINI_TFT_DMODEL", "32"))
    n_quantiles = int(os.getenv("MINI_TFT_NQUANTILES", "3"))

    m = MiniTFT(
        n_obs=n_obs,
        n_known=n_known,
        d_static=d_static,
        d_model=d_model,
        n_quantiles=n_quantiles
    ).to(_DEVICE)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state = torch.load(model_path, map_location=_DEVICE)
    m.load_state_dict(state)
    m.eval()
    _MODEL = m
    print(f"[INFO] Model loaded. n_obs={n_obs}, n_known={n_known}, n_quantiles={n_quantiles}")
    return _MODEL


def predict_from_arrays(obs, known, static):
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(_DEVICE)
    known_t = torch.from_numpy(known).unsqueeze(0).to(_DEVICE)
    static_t = torch.from_numpy(static).unsqueeze(0).to(_DEVICE)

    model = get_model()

    with torch.no_grad():
        pred_t, attn_t = model(obs_t, known_t, static_t)

    quantiles = pred_t.squeeze().cpu().numpy()
    median_prediction = float(quantiles[1])

    print(f"[INFO] Predictions - Q10: {quantiles[0]:.4f}, Q50: {quantiles[1]:.4f}, Q90: {quantiles[2]:.4f}")

    return median_prediction
