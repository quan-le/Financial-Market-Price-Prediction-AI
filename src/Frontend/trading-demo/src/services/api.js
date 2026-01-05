const API_BASE = process.env.REACT_APP_API_BASE;

export async function getModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error("Failed to load models");
  return res.json();
}

export async function getPrices(symbol, timeframe) {
  const res = await fetch(`${API_BASE}/prices?symbol=${symbol}&tf=${timeframe}`);
  if (!res.ok) throw new Error("Failed to load prices");
  return res.json();
}

export async function getPrediction(symbol, timeframe, model) {
  const res = await fetch(
    `${API_BASE}/predict?symbol=${symbol}&tf=${timeframe}&model=${model}`
  );
  if (!res.ok) throw new Error("Failed to load prediction");
  return res.json();
}
