import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  timeout: 30000,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:8000",
    headless: true,
    launchOptions: {
      executablePath: process.env.CHROMIUM_PATH || undefined,
      args: ["--no-sandbox", "--disable-dev-shm-usage"],
    },
    screenshot: "only-on-failure",
  },
  webServer: {
    command: `${process.env.PYTHON || "python"} -m uvicorn shopping_agent.app:app --host 127.0.0.1 --port 8000`,
    url: "http://127.0.0.1:8000/healthz",
    reuseExistingServer: false,
    env: {
      AUTH_MODE: "demo",
      REASONER_BACKEND: "rule",
      RETRIEVER_BACKEND: "baseline",
      SESSION_BACKEND: "memory",
    },
  },
});
