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

- `site/novelty/index.md` ("What we bring") is the reference: every claim of the project in one
  table, with its measurement, its evidence and where to try it. **Update it with every result.**
  `site/results/index.md` holds the numbers. An article does not replace either and does not
  restate them: it tells the story of one line of experiments and links to both.
- The Models catalogue (`site/models/`, the home page's "What works today") is driven by
  `site/.vitepress/theme/lib/catalogue.ts`. **Update it with every result, like What we bring**:
  a new model gets an entry, a verdict moves a card to its group, and every number on a card is
  copied from Results or a model card. `npm --prefix site test` checks it against the manifest.
- One Markdown file per article in `site/articles/`, in English like the rest of the site,
  listed on `site/articles/index.md`, newest first.
- An article is written for a wide technical audience: every term is explained the first time,
  every number says what it was measured on, and the limits of a result are stated as plainly as
  the result. Link to the evidence in the repository (pre-registrations in `docs/`, result files
  in `runs/`).

## Synthetic data and labels

- **Every generated example, label, paraphrase or teacher output that enters a training,
  validation or test set comes from an open-weight model whose licence allows that use**: today
  GLM-5.3-flash, through `sky-data teacher` (rules in `docs/s6-gen-prereg.md`, amendments 1 and 3).
  Never from a proprietary model or API whose terms restrict training on its outputs.
- Each generated source is listed in `DATA_LICENSES.md` with its model, its licence and its role.
- Text written for a demo or a test fixture (presets, edge cases) is not training data and never
  enters a data set.
