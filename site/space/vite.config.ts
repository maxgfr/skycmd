// One static demo app for every Hugging Face Space of the project (`npm run space:build`): the
// site's own Vue components (decision demos, proofs, games) with VitePress's `withBase` replaced
// by a shim. Each Space ships this build with its own `space.json`, model files and wasm module
// (scripts/hf-publish.py), the global Space (kind `hub`) too. The view of one model
// (SpaceView.vue) and the list of the global Space (HubView.vue) are chunks of their own, so the
// global Space's first screen loads no demo code.
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  base: './',
  plugins: [vue()],
  resolve: {
    // Only the bare `vitepress` import: the theme's CSS still comes from the package.
    alias: [{ find: /^vitepress$/, replacement: fileURLToPath(new URL('./vitepress-shim.ts', import.meta.url)) }],
  },
  build: {
    outDir: fileURLToPath(new URL('../../target/hf-space-app', import.meta.url)),
    emptyOutDir: true,
    target: 'es2022',
    chunkSizeWarningLimit: 900,
  },
})
