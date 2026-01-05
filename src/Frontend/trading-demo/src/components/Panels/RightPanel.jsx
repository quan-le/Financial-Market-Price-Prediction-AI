function RightPanel() {
  return (
    <div
      style={{
        backgroundColor: "#161a1e",
        borderRadius: "8px",
        padding: "12px",
        height: "100%",
      }}
    >
      <h4 style={{ marginBottom: "10px" }}>Markets</h4>
      <div style={{ color: "#848e9c", fontSize: "14px" }}>
        Waiting for backend market data…
      </div>

      <hr style={{ borderColor: "#2b3139", margin: "12px 0" }} />

      <h4 style={{ marginBottom: "10px" }}>AI Models</h4>
      <div style={{ color: "#848e9c", fontSize: "14px" }}>
        Model metadata will appear here
      </div>
    </div>
  );
}

export default RightPanel;
