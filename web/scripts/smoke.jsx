// SSR smoke test: prove the data-path components render without throwing on the
// real backtest.json. Bundled by esbuild + run in node (see verify below).
import React from "react";
import { renderToString } from "react-dom/server";
import data from "../public/backtest.json";
import { WinRateChart, ReturnChart, Dumbbell } from "../src/components/Charts.jsx";
import { StrategyTable } from "../src/components/StrategyTable.jsx";
import { TradesPanel } from "../src/components/TradesPanel.jsx";

const els = [
  React.createElement(WinRateChart, { rows: data.results }),
  React.createElement(ReturnChart, { rows: data.results, cfg: data.config }),
  React.createElement(Dumbbell, { rows: data.results }),
  React.createElement(StrategyTable, { rows: data.results }),
  React.createElement(TradesPanel, { results: data.results, trades: data.trades }),
];
const out = els.map((e) => renderToString(e)).join("\n");
for (const must of ["Validated", "Trade blotter", "Win rate by strategy", "Equity curve"]) {
  if (!out.includes(must)) throw new Error("render missing: " + must);
}
console.log("SMOKE OK — rendered", out.length, "chars, all sections present");
