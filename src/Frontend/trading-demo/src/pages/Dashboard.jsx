import { useEffect, useState } from "react";
import PriceChart from "../components/Chart/PriceChart";
import RightPanel from "../components/Panels/RightPanel";
import PredictionPanel from "../components/Panels/PredictionPanel";

function Dashboard() {
  // Controls (later map to backend query params)
  const [symbol, setSymbol] = useState("BTC / USD");
  const [timeframe, setTimeframe] = useState("1D");
  const [model, setModel] = useState("tft");

  // Data 
  const [candles, setCandles] = useState([]);
  const [prediction, setPrediction] = useState([]);

  // ENABLE_DEMO_DATA = false(Zero mock data))
    const ENABLE_DEMO_DATA = true;

useEffect(() => {
  if (!ENABLE_DEMO_DATA) return;

  const demoCandles = [
    { time: "2025-12-26", open: 61800, high: 62300, low: 61600, close: 62000, volume: 1200 },
    { time: "2025-12-27", open: 62000, high: 62450, low: 61900, close: 62100, volume: 1100 },
    { time: "2025-12-28", open: 62100, high: 62200, low: 61750, close: 61950, volume: 1400 },
    { time: "2025-12-29", open: 61950, high: 62380, low: 61820, close: 62200, volume: 1600 },
    { time: "2025-12-30", open: 62200, high: 62600, low: 62100, close: 62420, volume: 1500 },
  ];

  const demoPrediction = [
    { time: "2025-12-31", value: 62550 },
    { time: "2026-01-01", value: 62810 },
    { time: "2026-01-02", value: 62920 },
    { time: "2026-01-03", value: 62780 },
    { time: "2026-01-04", value: 63110 },
  ];

  setCandles(demoCandles);

  if (model === "minitft") {
    setPrediction(demoPrediction.map((p) => ({ ...p, value: Math.round(p.value * 0.997) })));
  } else {
    setPrediction(demoPrediction);
  }
}, [ENABLE_DEMO_DATA, model]);


  return (
    <div
      style={{
        backgroundColor: "#0b0e11",
        minHeight: "100vh",
        color: "white",
      }}
    >
      {/* Top Bar */}
      <div
        style={{
          padding: "16px",
          borderBottom: "1px solid #1e2329",
          display: "flex",
          flexDirection: "column",
          gap: "10px",
        }}
      >
        <div>
          <h2 style={{ margin: 0 }}>Financial Market Price Prediction AI</h2>
          <div style={{ color: "#848e9c", marginTop: "6px" }}>
            {symbol}
          </div>
        </div>

        {/* Controls */}
        <div
          style={{
            display: "flex",
            gap: "10px",
            flexWrap: "wrap",
            alignItems: "center",
          }}
        >
          <button
            onClick={() =>
              setSymbol((prev) =>
                prev === "BTC / USD" ? "ETH / USD" : "BTC / USD"
              )
            }
            style={{
              padding: "8px 12px",
              backgroundColor: "#1e2329",
              color: "white",
              border: "1px solid #2b3139",
              borderRadius: "6px",
              cursor: "pointer",
            }}
          >
            Switch to {symbol === "BTC / USD" ? "ETH" : "BTC"}
          </button>

          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            style={{
              padding: "8px 10px",
              backgroundColor: "#1e2329",
              color: "white",
              border: "1px solid #2b3139",
              borderRadius: "6px",
              cursor: "pointer",
            }}
          >
            <option value="1H">1H</option>
            <option value="4H">4H</option>
            <option value="1D">1D</option>
            <option value="1W">1W</option>
          </select>

          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            style={{
              padding: "8px 10px",
              backgroundColor: "#1e2329",
              color: "white",
              border: "1px solid #2b3139",
              borderRadius: "6px",
              cursor: "pointer",
            }}
          >
            <option value="tft">TFT</option>
            <option value="minitft">miniTFT</option>
          </select>

          <div style={{ color: "#848e9c", fontSize: "14px" }}>
            Demo data: {ENABLE_DEMO_DATA ? "ON" : "OFF"}
          </div>
        </div>
      </div>

      {/* Main Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "3fr 1fr",
          gap: "16px",
          padding: "16px",
        }}
      >
        <PriceChart
          symbol={symbol}
          timeframe={timeframe}
          model={model}
          candles={candles}
          prediction={prediction}
        />

        <RightPanel />
      </div>

      {/* Bottom Panel */}
      <div style={{ padding: "0 16px 16px" }}>
        <PredictionPanel prediction={prediction} />
      </div>
    </div>
  );
}

export default Dashboard;
