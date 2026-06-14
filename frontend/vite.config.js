import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api/auth': { target: 'http://localhost:4000', changeOrigin: true },
      '/api/admin': { target: 'http://localhost:4000', changeOrigin: true },
      '/api/upload': { target: 'http://localhost:8000', changeOrigin: true },
      '/api/scan': { target: 'http://localhost:8000', changeOrigin: true },
      '/api/scans': { target: 'http://localhost:8000', changeOrigin: true },
      '/api/download-report': { target: 'http://localhost:8000', changeOrigin: true },
      '/api/health': { target: 'http://localhost:8000', changeOrigin: true },
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
})
