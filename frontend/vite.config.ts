import { defineConfig } from "vite";
export default defineConfig({
  base: "/static/ui/",
  build: { outDir: "../src/shopping_agent/static/ui", emptyOutDir: true },
  server: {
    proxy: { "/chat": "http://127.0.0.1:8000", "/ui": "http://127.0.0.1:8000" },
  },
});
