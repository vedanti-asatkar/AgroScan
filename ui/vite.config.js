import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/ui/',
  server: {
    proxy: { '/health': 'http://localhost:8000', '/ready': 'http://localhost:8000', '/predict': 'http://localhost:8000', '/explain': 'http://localhost:8000', '/feedback': 'http://localhost:8000', '/dashboard': 'http://localhost:8000' },
  },
  build: { outDir: 'dist', emptyOutDir: true },
});
