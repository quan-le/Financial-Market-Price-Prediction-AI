import { useEffect, useRef } from "react";
import { createChart, CandlestickSeries, LineSeries } from "lightweight-charts";

function PriceChart(props) {
  const chartContainerRef = useRef(null);

  useEffect(() => {
    const container = chartContainerRef.current;
    if (!container) return;

    const chart = createChart(container, {
      width: container.clientWidth,
      height: 320,
      layout: {
        background: { color: "#161a1e" },
        textColor: "#d1d4dc",
      },
      grid: {
        vertLines: { color: "#2b3139" },
        horzLines: { color: "#2b3139" },
      },
      timeScale: {
        borderColor: "#2b3139",
      },
      rightPriceScale: {
        borderColor: "#2b3139",
      },
    });

    // 1) Price candles
    const candleSeries = chart.addSeries(CandlestickSeries);

    // 2) Prediction overlay line
    const predSeries = chart.addSeries(LineSeries, {
      lineWidth: 2,
    });

    // --- Set candle data (must be OHLC) ---
    const candles = (props.candles || []).map((c) => ({
      time: c.time, // "YYYY-MM-DD"
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    candleSeries.setData(candles);

    // --- Set prediction data (time/value) ---
    const prediction = (props.prediction || []).map((p) => ({
      time: p.time, // "YYYY-MM-DD" (often future dates)
      value: p.value,
    }));
    predSeries.setData(prediction);

    // Fit content so both series are visible
    chart.timeScale().fitContent();

    // Make chart responsive
    const handleResize = () => {
      chart.applyOptions({ width: container.clientWidth });
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [props.candles, props.prediction]);

  return (
    <div
      style={{
        backgroundColor: "#161a1e",
        borderRadius: "8px",
        padding: "12px",
        color: "#eaecef",
      }}
    >
      <div style={{ marginBottom: "8px", fontWeight: "bold" }}>
        {props.symbol} · {props.timeframe} · Model: {props.model}
      </div>

      <div style={{ color: "#848e9c", fontSize: "14px", marginBottom: "10px" }}>
        Candles: {props.candles?.length || 0} · Prediction points:{" "}
        {props.prediction?.length || 0}
      </div>

      <div ref={chartContainerRef} />
    </div>
  );
}

export default PriceChart;
