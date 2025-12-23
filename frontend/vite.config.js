import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Set base to repo name for GitHub Pages, e.g. '/dharma-visualizer/'
  // If using a custom domain or root, set to '/'
  base: './',
  server: {
    port: 3000
  }
})
