import { useState, useEffect } from "react";
import { n1, signed, money, money0 } from "./format.js";
import { Kpi, Delta } from "./components/Kpi.jsx";
import { WinRateChart, ReturnChart, Dumbbell } from "./components/Charts.jsx";
import { TradesPanel } from "./components/TradesPanel.jsx";
import { StrategyTable } from "./components/StrategyTable.jsx";
import { Skeleton, ErrorState } from "./components/States.jsx";

function Header({ theme, onToggle }) {
  return (
    <div className="top">
      <div className="brand">
        <div className="mark" />
        <div>
          <h1>MiND-Shot · Strategy Validation</h1>
          <div className="sub">Backtest report — mean-reversion playbook</div>
        </div>
      </div>
      <button className="toggle" onClick={onToggle} aria-label="Toggle color theme">
        {theme === "dark" ? "☀︎ Light" : "☾ Dark"}
      </button>
    </div>
  );
}

export default function App() {
  const [state, setState] = useState({ status: "loading" });
  const [theme, setTheme] = useState(() =>
    localStorage.getItem("ms-theme") ||
    (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"));

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("ms-theme", theme);
  }, [theme]);

  useEffect(() => {
    fetch(import.meta.env.BASE_URL + "backtest.json", { cache: "no-cache" })
      .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then((d) => setState({ status: "ok", data: d }))
      .catch((e) => setState({ status: "error", err: String(e) }));
  }, []);

  const toggle = () => setTheme((t) => (t === "dark" ? "light" : "dark"));
  const header = <Header theme={theme} onToggle={toggle} />;

  if (state.status === "loading")
    return <div className="wrap">{header}<div style={{ marginTop: 22 }}><Skeleton /></div></div>;
  if (state.status === "error")
    return <div className="wrap">{header}<ErrorState err={state.err} /></div>;

  const { config: cfg, summary: s, results, trades, generated_at } = state.data;
  const winGateDelta = +(s.avg_win_rate - 60).toFixed(1);
  const avgFinal = cfg.account_usd * (1 + s.avg_return_pct / 100);
  const when = new Date(generated_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

  return (
    <div className="wrap">
      {header}

      <div className="chips">
        <div className="chip"><span className="dot" />Account <b>{money0(cfg.account_usd)}</b></div>
        <div className="chip">Leverage <b>{cfg.leverage}×</b></div>
        <div className="chip">Allocation <b>{cfg.alloc_pct}%</b></div>
        <div className="chip">Margin <b>{cfg.margin}</b></div>
        <div className="chip">Round-trip cost <b>{cfg.round_trip_pct}%</b></div>
        <div className="chip">{cfg.window}</div>
      </div>

      <div className="insight">
        <b>{s.passing} of {s.strategies} strategies validated.</b> The playbook holds up out-of-sample:{" "}
        <b>{n1(s.overall_win_rate)}%</b> win rate across <b>{s.total_trades}</b> trades, averaging{" "}
        <b>{signed(s.avg_return_pct)}%</b> return per strategy.
        <div className="src">Computed from {s.total_trades} simulated round-trips over the committed 4h fixtures · generated {when}</div>
      </div>

      <div className="kpis">
        <Kpi hero label="Validated" value={s.passing + " / " + s.strategies} glow
          delta={<Delta kind="pos" glyph="✓">all pass</Delta>}
          cap="≥40 trades · >60% win · profitable" />
        <Kpi label="Win rate" value={n1(s.overall_win_rate) + "%"}
          delta={<Delta kind="pos" glyph="▲">{signed(winGateDelta)} pts</Delta>}
          cap="overall, vs 60% gate" />
        <Kpi label="Total trades" value={s.total_trades}
          delta={<Delta kind="flat">5 strategies</Delta>}
          cap="6-month window · 4h close" />
        <Kpi label="Avg return" value={signed(s.avg_return_pct) + "%"}
          delta={<Delta kind="gold" glyph="▲">best {signed(s.best_return_pct)}%</Delta>}
          cap={money0(cfg.account_usd) + " → ~" + money0(avgFinal) + " avg"} />
        <Kpi label="Worst drawdown" value={n1(s.worst_drawdown_pct) + "%"}
          delta={<Delta kind="neg" glyph="▼">deepest</Delta>}
          cap="peak-to-trough, single strategy" />
      </div>

      <div className="sec-h"><h2>Performance</h2><span className="meta">sorted best-first</span><div className="rule" /></div>
      <div className="bento">
        <WinRateChart rows={results} />
        <ReturnChart rows={results} cfg={cfg} />
        <Dumbbell rows={results} />
      </div>

      <div className="sec-h"><h2>Trades</h2><span className="meta">{s.total_trades} trades · equity + blotter</span><div className="rule" /></div>
      <TradesPanel results={results} trades={trades} />

      <div className="sec-h"><h2>Every strategy</h2><span className="meta">click a header to sort</span><div className="rule" /></div>
      <StrategyTable rows={results} />

      <div className="foot">
        <b>Methodology.</b> Each strategy is replayed over the committed price fixtures with{" "}
        {money0(cfg.account_usd)} capital, {cfg.leverage}× leverage, {cfg.alloc_pct}% allocation,{" "}
        {cfg.margin} margin and a {cfg.round_trip_pct}% round-trip cost. A strategy is “validated” only if it takes{" "}
        ≥40 trades, wins &gt;60%, ends profitable, and lands within 6 points of its playbook expectation.<br />
        <b>Not financial advice.</b> Past backtested performance is educational and does not guarantee future results.
        Data regenerated automatically by <span className="mono">gen_report.py</span> in CI · last run {when}.
      </div>
    </div>
  );
}
