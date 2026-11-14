import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const servicePort = process.env.PIXORA_PORT ?? '8732'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    // The UI only ever talks to the local service, through /api.
    proxy: { '/api': `http://127.0.0.1:${servicePort}` },
  },
})
