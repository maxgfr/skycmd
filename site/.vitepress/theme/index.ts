import DefaultTheme from 'vitepress/theme-without-fonts'
import type { Theme } from 'vitepress'
import { h } from 'vue'
import HeroGame from './components/HeroGame.vue'
import LiveStats from './components/LiveStats.vue'
import FeatureCards from './components/FeatureCards.vue'
import PlayGame from './components/PlayGame.vue'
import BenchView from './components/BenchView.vue'
import LiveDecide from './components/LiveDecide.vue'
import LiveProof from './components/LiveProof.vue'
import LiveCascade from './components/LiveCascade.vue'
import S5Results from './components/S5Results.vue'
import ForecastView from './components/ForecastView.vue'
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
    app.component('LiveDecide', LiveDecide)
    app.component('LiveProof', LiveProof)
    app.component('LiveCascade', LiveCascade)
    app.component('S5Results', S5Results)
    app.component('ForecastView', ForecastView)
  },
} satisfies Theme
