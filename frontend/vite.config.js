import { defineConfig } from 'vite'

export default defineConfig({
  // Build configuration
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    minify: 'esbuild', // Use esbuild instead of terser for faster builds
    rollupOptions: {
      input: {
        main: './index.html'
      }
    },
    // Disable chunk size warnings for production
    chunkSizeWarningLimit: 1000
  },
  
  // Development server configuration
  server: {
    port: 3000,
    host: true,
    cors: true
  },
  
  // Environment variables
  define: {
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'production')
  },
  
  // Base path for deployment
  base: './',
  
  // Plugins (add any Vite plugins if needed)
  plugins: []
})
