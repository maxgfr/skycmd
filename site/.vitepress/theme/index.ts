import DefaultTheme from 'vitepress/theme-without-fonts'
import type { Theme } from 'vitepress'
import { h } from 'vue'
import HeroGame from './components/HeroGame.vue'
import LiveStats from './components/LiveStats.vue'
import FeatureCards from './components/FeatureCards.vue'
import PlayGame from './components/PlayGame.vue'
import BenchView from './components/BenchView.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout: () =>
    h(DefaultTheme.Layout, null, {
      // The live game takes the place of the hero image on the landing page.
      'home-hero-image': () => h(HeroGame),
    }),
  enhanceApp({ app }) {
    app.component('LiveStats', LiveStats)
    app.component('FeatureCards', FeatureCards)
    app.component('PlayGame', PlayGame)
    app.component('BenchView', BenchView)
  },
} satisfies Theme
