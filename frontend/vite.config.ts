import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', 'DASHBOARD_')
  const distributed = env.DASHBOARD_DISTRIBUTED_URL || 'http://127.0.0.1:9210'
  return {
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api/distributed': { target: distributed, changeOrigin: true },
      '/ws/distributed': { target: distributed, ws: true },
      '/api': {
        target: env.DASHBOARD_LOCAL_URL || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: env.DASHBOARD_LOCAL_URL || 'http://127.0.0.1:8000',
        ws: true,
      },
    },
  },
}
})
