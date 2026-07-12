import { useState, useMemo } from "react";
import { signed, money } from "../format.js";
import { EquityCurve } from "./EquityCurve.jsx";

const COLORS = ["var(--c0)", "var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)"];
const fmtDate = (ts) => new Date(ts * 1000).toLocaleDateString(undefined, { month: "short", day: "2-digit", year: "2-digit" });

export function TradesPanel({ results, trades }) {
  const [sel, setSel] = useState("all");
  const colorOf = useMemo(() => Object.fromEntries(results.map((r, i) => [r.strategy_id, COLORS[i % COLORS.length]])), [results]);

  const series = sel === "all"
    ? results.map((r) => ({ id: r.strategy_id, name: r.name, color: colorOf[r.strategy_id], points: r.equity }))
    : (() => { const r = results.find((x) => x.strategy_id === sel); return [{ id: r.strategy_id, name: r.name, color: colorOf[r.strategy_id], points: r.equity }]; })();

  const rows = useMemo(() => {
    const list = sel === "all" ? trades : trades.filter((t) => t.strategy_id === sel);
    return [...list].sort((a, b) => b.exit_ts - a.exit_ts);
  }, [sel, trades]);

  const wins = rows.filter((t) => t.win).length;
  const wr = rows.length ? Math.round((wins / rows.length) * 1000) / 10 : 0;

  return (
    <div>
      <div className="tabs" role="tablist" aria-label="Filter trades by strategy">
        <button className="tab" role="tab" aria-pressed={sel === "all"} onClick={() => setSel("all")}>
          All strategies
        </button>
        {results.map((r) => (
          <button key={r.strategy_id} className="tab" role="tab" aria-pressed={sel === r.strategy_id}
            onClick={() => setSel(r.strategy_id)} title={r.name}>
            <span className="lg-dot" style={{ background: colorOf[r.strategy_id], display: "inline-block", marginRight: 6, verticalAlign: "middle" }} />
            {r.name}
          </button>
        ))}
      </div>

      <div className="bento">
        <div className="card span12">
          <h3>Equity curve — {money(100)} start{sel === "all" ? " · all five accounts" : ""}</h3>
          <div className="ch-sub">
            Wallet balance after each closed trade. Dashed line = starting capital.
            {sel === "all" ? " Each strategy runs its own independent $100 account." : ""}
          </div>
          <EquityCurve series={series} baseline={100} />
          {sel === "all" && (
            <div className="legend">
              {results.map((r) => (
                <span key={r.strategy_id}>
                  <i className="lg-line" style={{ background: colorOf[r.strategy_id] }} />{r.name}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="card span12">
          <h3>Trade blotter</h3>
          <div className="ch-sub">
            {rows.length} trade{rows.length === 1 ? "" : "s"} · {wins} wins · {wr}% win rate — most recent first.
          </div>
          <div className="tbl-wrap scroll">
            <table>
              <thead>
                <tr>
                  <th>Closed</th>
                  {sel === "all" && <th>Strategy</th>}
                  <th>Coin</th>
                  <th>Side</th>
                  <th className="num">Entry</th>
                  <th className="num">Exit</th>
                  <th className="num">P&L</th>
                  <th className="num">Return</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((t, i) => (
                  <tr key={t.strategy_id + i}>
                    <td className="mono" style={{ color: "var(--text-muted)", fontSize: ".78rem" }}>{fmtDate(t.exit_ts)}</td>
                    {sel === "all" && <td style={{ color: "var(--text-muted)" }}>{t.name}</td>}
                    <td><span className="coin">{t.asset}</span></td>
                    <td><span className={"side " + t.side}>{t.side === "long" ? "▲ Long" : "▼ Short"}</span></td>
                    <td className="num">{t.entry}</td>
                    <td className="num">{t.exit}</td>
                    <td className={"num pl " + (t.pnl_usd >= 0 ? "pos" : "neg")}>{(t.pnl_usd >= 0 ? "+" : "−") + "$" + Math.abs(t.pnl_usd).toFixed(2)}</td>
                    <td className={"num pl " + (t.ret_pct >= 0 ? "pos" : "neg")}>{signed(t.ret_pct)}%</td>
                    <td><span className={"res " + (t.win ? "win" : "loss")}>{t.win ? "Win" : "Loss"}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
