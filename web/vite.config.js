import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base must match the GitHub Pages repo path: https://<user>.github.io/MiND-Shot/
export default defineConfig({
  base: "/MiND-Shot/",
  plugins: [react()],
  build: { outDir: "dist", sourcemap: false },
});
