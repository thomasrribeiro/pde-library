import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig(({ mode }) => ({
    base: mode === 'production' ? '/pde-library/' : '/',
    build: {
        outDir: 'dist',
        assetsDir: 'assets',
        rollupOptions: {
            input: {
                main: resolve(__dirname, 'index.html')
            }
        }
    },
    server: {
        fs: {
            // Allow serving files from parent directory (benchmarks/)
            allow: ['..']
        }
    },
    assetsInclude: ['**/*.npz']
}));
