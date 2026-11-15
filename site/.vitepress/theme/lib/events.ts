// Step events of the game cores (`sky_games::Event`, snake_case in the session's JSON), read by
// the page's counters. No imports: the site's Node tests load it directly.

/** The step event that ends an episode in a crash (`Event::Crashed`; doom: `Event::Died`). */
export function endedInCrash(event: string): boolean {
  return event === 'crashed' || event === 'died'
}
