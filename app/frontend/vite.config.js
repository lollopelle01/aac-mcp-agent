import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,   // bind su 0.0.0.0 — raggiungibile sia da localhost che dalla LAN (tablet)
    port: 5173,
    proxy: {
      // Forward all /api/* calls to the FastAPI backend during dev
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
