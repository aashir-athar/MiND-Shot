// Small display helpers — kept pure so they're trivially testable.
export const n1 = (x) => (x == null ? "—" : x.toFixed(1));
export const signed = (x) =>
  (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(1);
export const money = (x) => "$" + x.toFixed(2);
export const money0 = (x) => "$" + Math.round(x);
