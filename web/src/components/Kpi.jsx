export function Delta({ kind, glyph, children }) {
  return (
    <span className={"delta " + kind}>
      {glyph ? glyph + " " : ""}
      {children}
    </span>
  );
}

export function Kpi({ label, value, glow, delta, cap, hero }) {
  return (
    <div className={"kpi" + (hero ? " hero" : "")}>
      <div className="label">{label}</div>
      <div className={"val mono" + (glow ? " glow" : "")}>{value}</div>
      {delta || null}
      <div className="cap">{cap}</div>
    </div>
  );
}
