import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  test: {
    environment: 'jsdom',
    env: { VITE_API_BASE_URL: process.env.VITE_API_BASE_URL || 'http://localhost:8000' },
    setupFiles: ['./src/test/setup.js'],
    testTimeout: 180000,
    hookTimeout: 60000,
  },
})
