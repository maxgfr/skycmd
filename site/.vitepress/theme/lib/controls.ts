// Human controls of the Play page: keys and swipes to action keys, and the first press made in AI
// mode. No imports: the site's Node tests load it directly.

export type ControlGame = 'flappy' | 'rescue' | 'dodge' | 'doom'

export const KEYMAP: Record<ControlGame, Record<string, string>> = {
  flappy: { ' ': 'thrust', ArrowUp: 'thrust', w: 'thrust' },
  rescue: {
    ArrowUp: 'north',
    w: 'north',
    ArrowDown: 'south',
    s: 'south',
    ArrowRight: 'east',
    d: 'east',
    ArrowLeft: 'west',
    a: 'west',
    h: 'hover',
    r: 'return_home',
  },
  dodge: {
    ArrowUp: 'over',
    ArrowLeft: 'around',
    ArrowRight: 'around',
    ArrowDown: 'brake',
    ' ': 'continue',
    a: 'abort',
  },
  doom: {
    w: 'forward',
    ArrowUp: 'forward',
    s: 'back',
    ArrowDown: 'back',
    a: 'strafe_left',
    d: 'strafe_right',
    ArrowLeft: 'turn_left',
    ArrowRight: 'turn_right',
    q: 'turn_left',
    ' ': 'shoot',
    Control: 'shoot',
    e: 'use',
  },
}

/** The action key of a keyboard key (`KeyboardEvent.key`), if the game maps it. */
export function actionForKey(kind: ControlGame, key: string): string | undefined {
  return KEYMAP[kind][key.length === 1 ? key.toLowerCase() : key]
}

/** Swipes shorter than this (px) are taps. */
export const SWIPE_MIN = 24

/** The action of a swipe on the canvas (rescue: a move, dodge: a maneuver); none for a tap or a
 * game without swipe controls. Screen y grows downward. */
export function actionForSwipe(kind: ControlGame, dx: number, dy: number): string | undefined {
  if (Math.hypot(dx, dy) < SWIPE_MIN) return undefined
  if (kind === 'rescue') return Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'east' : 'west') : dy > 0 ? 'south' : 'north'
  if (kind === 'dodge') return Math.abs(dx) > Math.abs(dy) ? 'around' : dy < 0 ? 'over' : 'brake'
  return undefined
}

/**
 * A control used in AI mode switches the page to human mode, which starts a new runner: the
 * press is kept here and replayed on the new runner, so the first press also acts.
 */
export class FirstPress {
  private key: string | null = null

  /** A control was used in `mode`: true when the page must switch to human mode first (the
   * press is then kept for [`take`]); false when it can go to the runner now. */
  press(mode: 'ai' | 'human', key: string): boolean {
    if (mode === 'human') return false
    this.key = key
    return true
  }

  /** The press to replay on the new human runner (once). */
  take(): string | null {
    const k = this.key
    this.key = null
    return k
  }
}
