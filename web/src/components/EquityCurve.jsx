import { money0 } from "../format.js";

// Responsive SVG line chart. `series`: [{ id, name, color, points:[wallet...] }].
// Single series gets an area fill + end dot; multiple series draw as overlaid lines.
export function EquityCurve({ series, baseline = 100 }) {
  const W = 640, H = 220, PL = 44, PR = 14, PT = 14, PB = 24;
  const maxLen = Math.max(...series.map((s) => s.points.length), 2);
  const all = series.flatMap((s) => s.points).concat(baseline);
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.08 || 1;
  lo -= pad; hi += pad;

  const x = (i) => PL + (i / (maxLen - 1)) * (W - PL - PR);
  const y = (v) => PT + (1 - (v - lo) / (hi - lo)) * (H - PT - PB);
  const line = (pts) => pts.map((v, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1)).join(" ");
  const single = series.length === 1;
  const ticks = [lo + (hi - lo) * 0.25, lo + (hi - lo) * 0.5, lo + (hi - lo) * 0.75, hi - pad * 0.5];

  return (
    <div className="eq">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Equity curve over the trade sequence" preserveAspectRatio="none">
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={PL} x2={W - PR} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth="1" />
            <text className="axlbl" x={PL - 6} y={y(t) + 3} textAnchor="end">{money0(t)}</text>
          </g>
        ))}
        {/* starting-capital baseline */}
        <line x1={PL} x2={W - PR} y1={y(baseline)} y2={y(baseline)} stroke="var(--text-faint)" strokeWidth="1" strokeDasharray="4 4" opacity="0.6" />
        {single && (
          <>
            <defs>
              <linearGradient id="eqfill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={series[0].color} stopOpacity="0.28" />
                <stop offset="100%" stopColor={series[0].color} stopOpacity="0" />
              </linearGradient>
            </defs>
            <path d={`${line(series[0].points)} L ${x(series[0].points.length - 1)} ${y(lo)} L ${x(0)} ${y(lo)} Z`} fill="url(#eqfill)" />
          </>
        )}
        {series.map((s) => (
          <path key={s.id} d={line(s.points)} fill="none" stroke={s.color} strokeWidth={single ? 2.2 : 1.8}
            strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {single && (
          <circle cx={x(series[0].points.length - 1)} cy={y(series[0].points.at(-1))} r="3.5"
            fill={series[0].color} stroke="var(--surface-1)" strokeWidth="1.5" />
        )}
      </svg>
    </div>
  );
}
