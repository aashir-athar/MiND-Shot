// Render smoke test using Vite's SSR loader (transforms JSX on the fly).
import { createServer } from "vite";

const server = await createServer({
  root: process.cwd(),
  server: { middlewareMode: true },
  appType: "custom",
  logLevel: "error",
  optimizeDeps: { noDiscovery: true },
});
try {
  await server.ssrLoadModule("/scripts/smoke.jsx"); // runs the render + assertions at import time
} finally {
  await server.close();
}
