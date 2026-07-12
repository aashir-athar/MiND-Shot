import { useState, useMemo } from "react";
import { n1, signed, money } from "../format.js";

const COLS = [
  { k: "name", t: "Strategy", num: false },
  { k: "asset", t: "Coin", num: false },
  { k: "win_rate", t: "Win %", num: true },
  { k: "expected_win_rate", t: "Exp %", num: true },
  { k: "delta", t: "Δ", num: true },
  { k: "trades", t: "Trades", num: true },
  { k: "ls", t: "L / S", num: true },
  { k: "final_wallet", t: "Final $", num: true },
  { k: "return_pct", t: "Return %", num: true },
  { k: "max_drawdown_pct", t: "Max DD %", num: true },
  { k: "passes", t: "Verdict", num: false },
];

const val = (r, k) =>
  k === "delta" ? r.win_rate - r.expected_win_rate : k === "ls" ? r.long_trades : r[k];

export function StrategyTable({ rows }) {
  const [sort, setSort] = useState({ k: "win_rate", dir: -1 });
  const data = useMemo(() => {
    return [...rows].sort((a, b) => {
      const x = val(a, sort.k), y = val(b, sort.k);
      return typeof x === "string" ? x.localeCompare(y) * sort.dir : (x - y) * sort.dir;
    });
  }, [rows, sort]);
  const click = (k) => setSort((s) => ({ k, dir: s.k === k ? -s.dir : -1 }));

  return (
    <div className="tbl-wrap">
      <table>
        <thead>
          <tr>
            {COLS.map((c) => (
              <th key={c.k} className={(c.num ? "num " : "") + "sortable"}
                aria-sort={sort.k === c.k ? (sort.dir === 1 ? "ascending" : "descending") : "none"}
                onClick={() => click(c.k)}>
                {c.t}{sort.k === c.k && <span className="arw">{sort.dir === 1 ? "▲" : "▼"}</span>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((r) => {
            const d = r.win_rate - r.expected_win_rate;
            return (
              <tr key={r.strategy_id}>
                <td className="st-name">{r.name}</td>
                <td><span className="coin">{r.asset}</span></td>
                <td className="num" style={{ color: "var(--brand-bright)", fontWeight: 600 }}>{n1(r.win_rate)}</td>
                <td className="num" style={{ color: "var(--text-muted)" }}>{n1(r.expected_win_rate)}</td>
                <td className="num" style={{ color: Math.abs(d) <= 6 ? "var(--pos)" : "var(--amber)" }}>{signed(d)}</td>
                <td className="num">{r.trades}</td>
                <td className="num" style={{ color: "var(--text-muted)" }}>{r.long_trades} / {r.short_trades}</td>
                <td className="num">{money(r.final_wallet)}</td>
                <td className="num" style={{ color: r.return_pct >= 0 ? "var(--pos)" : "var(--neg)" }}>{signed(r.return_pct)}</td>
                <td className="num" style={{ color: "var(--amber)" }}>{n1(r.max_drawdown_pct)}</td>
                <td>{r.passes
                  ? <span className="badge ok">✓ Validated</span>
                  : <span className="badge no">✗ Failed</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
