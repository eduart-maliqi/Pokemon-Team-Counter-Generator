import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Standard-Vite-Setup mit React. Dev-Server laeuft auf Port 5173,
// den die API (CORS) freigegeben hat.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
