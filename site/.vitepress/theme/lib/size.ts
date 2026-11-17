// File sizes as the repository writes them (README, BENCH.md, the cards, the Models catalogue):
// decimal units, KB = 1,000 bytes and MB = 1,000,000 bytes. Under 1 MB a whole number of KB,
// from 1 MB on one decimal of MB: 631,936 bytes is "632 KB", 2,418,912 bytes "2.4 MB". Every
// place of the site and of the Spaces that prints a byte count goes through here, so the same
// file reads the same everywhere. Pure: no dependency, tested by tests/size.test.ts.

export type SizeUnit = 'KB' | 'MB'

/** The number and its unit apart, for a layout that styles the unit on its own. */
export function sizeParts(bytes: number): { value: string; unit: SizeUnit } {
  const kb = Math.round(bytes / 1e3)
  if (kb < 1000) return { value: kb.toLocaleString('en-US'), unit: 'KB' }
  return {
    value: (bytes / 1e6).toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 }),
    unit: 'MB',
  }
}

/** "632 KB", "2.4 MB", "20.7 MB". */
export function formatSize(bytes: number): string {
  const { value, unit } = sizeParts(bytes)
  return `${value} ${unit}`
}
