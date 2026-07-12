export function Skeleton() {
  return (
    <div>
      <div className="kpis">
        {[0, 1, 2, 3, 4].map((i) => (
          <div className="kpi" key={i}>
            <div className="sk" style={{ height: "12px", width: "50%" }} />
            <div className="sk" style={{ height: "30px", width: "70%", marginTop: "8px" }} />
            <div className="sk" style={{ height: "10px", width: "85%", marginTop: "auto" }} />
          </div>
        ))}
      </div>
      <div className="bento" style={{ marginTop: "38px" }}>
        <div className="card span7"><div className="sk" style={{ height: "220px" }} /></div>
        <div className="card span5"><div className="sk" style={{ height: "220px" }} /></div>
      </div>
    </div>
  );
}

export function ErrorState({ err }) {
  return (
    <div className="err">
      <b>Couldn't load the report.</b>
      <div style={{ color: "var(--text-muted)", marginTop: "8px" }}>
        Expected <span className="mono">backtest.json</span> next to this page. ({err})
      </div>
    </div>
  );
}
