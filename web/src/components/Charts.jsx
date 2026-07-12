import { n1, signed } from "../format.js";

export function BarRow({ name, pct, max, label, gold, gate }) {
  const w = Math.max(0, Math.min(100, (pct / max) * 100));
  const inside = w > 22;
  return (
    <div className="row">
      <div className="name" title={name}>{name}</div>
      <div className="bar-track">
        <div className={"bar-fill" + (gold ? " gold" : "")} style={{ width: w + "%" }} />
        {gate != null && <div className="gate" style={{ left: (gate / max) * 100 + "%" }} />}
        <div className="bar-val" style={inside ? { right: 100 - w + "%" } : { left: w + "%" }}>{label}</div>
      </div>
    </div>
  );
}

export function WinRateChart({ rows }) {
  const sorted = [...rows].sort((a, b) => b.win_rate - a.win_rate);
  return (
    <div className="card span7">
      <h3>Win rate by strategy</h3>
      <div className="ch-sub">Gold line marks the 60% validation gate — every strategy clears it.</div>
      {sorted.map((r) => (
        <BarRow key={r.strategy_id} name={r.name} pct={r.win_rate} max={100}
          label={n1(r.win_rate) + "%"} gate={60} />
      ))}
      <div style={{ textAlign: "right", marginTop: "4px" }}>
        <span className="gate-lbl">▏ 60% pass gate</span>
      </div>
    </div>
  );
}

export function ReturnChart({ rows, cfg }) {
  const max = Math.max(...rows.map((r) => r.return_pct), 1);
  const sorted = [...rows].sort((a, b) => b.return_pct - a.return_pct);
  return (
    <div className="card span5">
      <h3>Return on ${cfg.account_usd.toFixed(0)}</h3>
      <div className="ch-sub">Net of {cfg.round_trip_pct}% round-trip cost, {cfg.leverage}× on {cfg.alloc_pct}% allocation.</div>
      {sorted.map((r) => (
        <BarRow key={r.strategy_id} name={r.name} pct={r.return_pct} max={max}
          label={signed(r.return_pct) + "%"} gold />
      ))}
    </div>
  );
}

export function Dumbbell({ rows, lo = 55, hi = 85 }) {
  const clamp = (v) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100));
  return (
    <div className="card span12">
      <h3>Backtest vs playbook expectation</h3>
      <div className="ch-sub">
        How tightly the out-of-sample win rate tracks the published playbook — ✓ = within the 6-point tolerance band.
      </div>
      {rows.map((r) => {
        const a = clamp(r.win_rate), e = clamp(r.expected_win_rate);
        const d = r.win_rate - r.expected_win_rate;
        const ok = Math.abs(d) <= 6;
        return (
          <div className="db" key={r.strategy_id}>
            <div className="name" title={r.name}>{r.name}</div>
            <div className="db-track">
              <div className="db-line" style={{ left: Math.min(a, e) + "%", width: Math.abs(a - e) + "%" }} />
              <div className="db-dot exp" style={{ left: e + "%" }} title={"expected " + n1(r.expected_win_rate) + "%"} />
              <div className="db-dot act" style={{ left: a + "%" }} title={"actual " + n1(r.win_rate) + "%"} />
            </div>
            <div className="d mono" style={{ color: ok ? "var(--pos)" : "var(--amber)" }}>
              {(ok ? "✓ " : "") + signed(d)}
            </div>
          </div>
        );
      })}
      <div className="legend">
        <span><i className="lg-dot" style={{ background: "var(--surface-2)", border: "2px solid var(--text-faint)" }} />Playbook expected</span>
        <span><i className="lg-dot" style={{ background: "var(--brand)", border: "2px solid var(--brand-bright)" }} />Backtest actual</span>
      </div>
    </div>
  );
}
