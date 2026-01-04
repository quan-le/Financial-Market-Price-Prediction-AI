function PredictionPanel({ prediction }) {
  if (!prediction || prediction.length === 0) {
    return (
      <div
        style={{
          backgroundColor: "#161a1e",
          borderRadius: "8px",
          padding: "12px",
          color: "#848e9c",
        }}
      >
        No predictions yet — select a model and timeframe.
      </div>
    );
  }

  return (
    <div
      style={{
        backgroundColor: "#161a1e",
        borderRadius: "8px",
        padding: "12px",
      }}
    >
      <h4>Prediction Table</h4>
      <table style={{ width: "100%", fontSize: "14px" }}>
        <thead>
          <tr>
            <th align="left">Date</th>
            <th align="right">Predicted Price</th>
          </tr>
        </thead>
        <tbody>
          {prediction.map((p) => (
            <tr key={p.time}>
              <td>{p.time}</td>
              <td align="right">{p.value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default PredictionPanel;
