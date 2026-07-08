import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/dharma_visualizer/',
  server: {
    port: 3000
  }
})
