import { defineConfig } from 'vite'

export default defineConfig({
  // Build configuration
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    minify: 'terser',
    rollupOptions: {
      input: {
        main: './index.html'
      }
    }
  },
  
  // Development server configuration
  server: {
    port: 3000,
    host: true,
    cors: true
  },
  
  // Environment variables
  define: {
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development')
  },
  
  // Base path for deployment
  base: './',
  
  // Plugins (add any Vite plugins if needed)
  plugins: []
})
