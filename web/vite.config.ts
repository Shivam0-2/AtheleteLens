import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', 'VITE_')
  const apiProxyTarget = env.VITE_API_PROXY_TARGET?.trim() || 'http://127.0.0.1:8000'
  const apiProxy = {
    '/api': {
      target: apiProxyTarget,
      changeOrigin: true,
    },
  }

  return {
    plugins: [react(), tailwindcss()],
    server: {
      host: '0.0.0.0',
      port: 5173,
      strictPort: true,
      proxy: apiProxy,
    },
    preview: {
      host: '0.0.0.0',
      port: 4173,
      strictPort: true,
      proxy: apiProxy,
    },
  }
})
