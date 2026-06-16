import { defineConfig } from "vitest/config";
import path from "node:path";

// Minimal, isolated config for unit tests — deliberately NOT the main
// vite.config (which pulls in the React plugin + wasm/worker handling that
// pure-logic tests don't need). Just the `@` alias and a node environment.
export default defineConfig({
    resolve: {
        alias: { "@": path.resolve(__dirname, "src") },
    },
    test: {
        environment: "node",
        include: ["src/**/*.test.ts"],
    },
});
