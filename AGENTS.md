# Instructions for agents working on skycmd

## Where things are published

- **Everything about skycmd is published on skycmd's own site**: the VitePress site in `site/`,
  deployed to <https://maxgfr.github.io/skycmd/> on every push to `main`. That covers results,
  model pages, demos, tutorials and **articles** (`site/articles/`).
- **Never publish skycmd content on the owner's personal site** (the repository
  `maxgfr/maxgfr.github.io`, served at <https://maxgfr.github.io/>). That site is the owner's:
  touch it only when the owner names it in the request.
- Models and their demos also go to Hugging Face (`scripts/hf-publish.py`, `hf/cards/`,
  `hf/spaces.json`); a result is published whether its gate passed or failed.

## Articles

- One Markdown file per article in `site/articles/`, in English like the rest of the site,
  listed on `site/articles/index.md`, newest first.
- An article is written for a wide technical audience: every term is explained the first time,
  every number says what it was measured on, and the limits of a result are stated as plainly as
  the result. Link to the evidence in the repository (pre-registrations in `docs/`, result files
  in `runs/`).
