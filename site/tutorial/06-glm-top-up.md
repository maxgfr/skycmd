# 06 GLM top-up via z.ai

Some data does not exist on the Hub in the form skycmd needs: French paraphrases of decision questions, and civil drone orders paired with valid command plans. `sky-data teacher` generates both with a GLM model behind z.ai's Anthropic-compatible endpoint. Every reply has to pass strict parsing and coded validation before it is written. This chapter covers the client, its budget controls and the two teacher tasks.

## What the teacher is for

| task | input | output |
|---|---|---|
| `export-questions` | decision JSONL in `data/decisions`, `data/games` | `data/teacher/questions.json` (unique questions) |
| `paraphrase-questions` | `questions.json` | `data/teacher/paraphrases.jsonl` (EN + FR rewrites) |
| `drone-orders` | none (themes are built in) | `data/teacher/orders.jsonl` plus `orders_decisions.jsonl` |

The game questions do not need the teacher. Their English and French paraphrases are written by hand in each game's `QUESTIONS` table ([chapter 04](./04-game-cores-and-oracles)). The teacher covers the Hub-derived questions, which mostly come in one fixed English template per source.

## Endpoint, key and model

Configuration is resolved at runtime, first from environment variables, then from the `ccs` CLI's `zai` profile:

`crates/sky-data/src/teacher.rs`

```rust
    pub fn resolve(args: &TeacherArgs) -> Result<Self> {
        let base_url = env_nonempty("SKY_TEACHER_URL")
            .or_else(|| ccs_get("base_url"))
            .unwrap_or_else(|| "https://api.z.ai/api/anthropic".into())
            .trim_end_matches('/')
            .to_string();
        let api_key = env_nonempty("SKY_TEACHER_KEY")
            .or_else(|| ccs_get("api_key"))
            .ok_or_else(|| anyhow!("no teacher API key: set SKY_TEACHER_KEY or configure `ccs config ... zai api_key`"))?;
        let model = args
            .model
            .clone()
            .or_else(|| env_nonempty("SKY_TEACHER_MODEL"))
            .or_else(|| ccs_get("model"))
            .unwrap_or_else(|| "auto".into());
```

| variable | default |
|---|---|
| `SKY_TEACHER_URL` | `https://api.z.ai/api/anthropic` |
| `SKY_TEACHER_KEY` | none, required (or `ccs config get zai api_key`) |
| `SKY_TEACHER_MODEL` | `auto` (overridden by `--model`) |

The key is a private field of `TeacherConfig`, and the struct's `Debug` implementation prints `<redacted>` in its place. A unit test (`debug_redacts_key`) checks this. The key is only ever sent as the `x-api-key` header and is never logged or written to disk.

With `auto`, the client calls `GET /v1/models` and picks the newest full GLM model, ranking variants named `air`, `flash`, `turbo`, `mini` or `lite` below it:

`crates/sky-data/src/teacher.rs`

```rust
pub fn choose_model(list: &Value) -> Option<String> {
    let mut models: Vec<(String, String)> = list
        .get("data")?
        .as_array()?
        .iter()
        .filter_map(|m| Some((m.get("id")?.as_str()?.to_string(), m.get("created_at").and_then(Value::as_str).unwrap_or("").to_string())))
        .filter(|(id, _)| id.to_ascii_lowercase().contains("glm"))
        .collect();
    let lite = |id: &str| ["air", "flash", "turbo", "mini", "lite"].iter().any(|w| id.to_ascii_lowercase().contains(w));
    models.sort_by(|a, b| (!lite(&a.0), &a.1, &a.0).cmp(&(!lite(&b.0), &b.1, &b.0)));
    models.pop().map(|m| m.0)
}
```

If the model list cannot be fetched, the client falls back to `DEFAULT_MODEL = "glm-4.6"`. If your plan only allows certain models, pass `--model` explicitly.

## Budget, pacing and retries

A teacher run can be expensive, so the client enforces three limits:

- `--max-calls` (default 300): a hard cap on HTTP requests. Retries count too.
- `--min-interval-ms` (default 1500): the minimum delay between two requests.
- `--batch` (default 40): items per request.

`crates/sky-data/src/teacher.rs`

```rust
    fn post(&mut self, body: &Value) -> Result<(u16, Value)> {
        if self.calls >= self.cfg.max_calls {
            return Err(BudgetExhausted.into());
        }
        if let Some(last) = self.last {
            let wait = self.cfg.min_interval.saturating_sub(last.elapsed());
            sleep(wait);
        }
        self.calls += 1;
        self.last = Some(Instant::now());
```

`call_json` makes up to 4 attempts per prompt, backing off exponentially. It retries network errors, HTTP 429, HTTP 5xx and replies that are not valid JSON, and fails immediately on any other status. Requests use the Anthropic Messages format (`/v1/messages`, `anthropic-version: 2023-06-01`), with `temperature` 0.9, `max_tokens` 16000 and thinking disabled. At the end of a run, the CLI prints the calls used and the input and output token totals.

All retries draw on the same `--max-calls` budget. A 429 caused by a plan restriction, rather than by rate limiting, will not clear on retry, and each failing batch costs up to 4 calls. If the log shows repeated `HTTP 429` errors with the same message, stop the run and fix the model or the plan before retrying.

## Strict JSON only

The teacher must answer with one JSON value and nothing else. A single Markdown code fence is tolerated. Leading prose or trailing text fails the reply:

`crates/sky-data/src/teacher.rs`

```rust
pub fn parse_strict_json(text: &str) -> Result<Value> {
    let t = text.trim();
    let t = if let Some(rest) = t.strip_prefix("```") {
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        rest.strip_suffix("```").ok_or_else(|| anyhow!("unterminated code fence"))?.trim()
    } else {
        t
    };
    serde_json::from_str(t).map_err(|e| anyhow!("reply is not strict JSON: {e}"))
}
```

The client never searches for JSON inside prose. A reply that needs to be repaired is a reply that should not be trusted.

## Task 1: question paraphrases

`export-questions` collects unique `(type, instructions, options)` triples from non-test records, shuffles them with the global seed, and keeps `--max` of them (default 2000). Test records are skipped, so the external benchmark never leaks into the teacher prompts.

`paraphrase-questions` asks for `--n` English and `--n` French rewrites per question (default 3). The prompt requires the same meaning, the same question type and the same numbers, units and named entities. Each reply item is checked against its batch:

`crates/sky-data/src/teacher.rs`

```rust
        let take = |k: &str| -> Option<Vec<String>> {
            let v: Vec<String> = it
                .get(k)?
                .as_array()?
                .iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && s.chars().count() <= 400 && s != &q.ins)
                .collect::<Vec<_>>();
```

Unknown ids are ignored, as are empty strings, strings over 400 characters, and copies of the original text. Duplicates are removed. Results are appended to the output file, and question ids already present are skipped on the next run, so an interrupted run resumes where it stopped.

## Task 2: civil drone orders

`drone-orders` asks for spoken orders, in about 50% French, that a civil operator could give. Each order comes with the plan the drone should execute. Six categories are allowed:

`crates/sky-data/src/teacher.rs`

```rust
pub const CATEGORIES: [&str; 6] = ["inspection", "mapping", "agriculture", "search_rescue", "ambiguous", "unsafe"];
```

Each request picks 5 of 24 built-in civil themes, such as bridge pillars, wind turbine blades, a vineyard or a flooded village. An `ambiguous` order lacks essential information, and its plan must be a single `clarify` action. An `unsafe` order is anything outside civil drone work, and its plan must be a single `refuse` action. The model therefore learns that asking and declining are normal answers.

### Strict plan validation

`crates/sky-data/src/plan.rs` mirrors the `sky-schema` plan format, but it is stricter than the runtime validator. It rejects plans and never corrects them: unknown actions, missing or extra fields, wrong types and out-of-range values all fail.

`crates/sky-data/src/plan.rs`

```rust
pub fn validate_plan(v: &Value) -> Result<Vec<String>> {
    let list = match v {
        Value::Object(m) if m.len() == 1 => m.get("plan").and_then(Value::as_array),
        Value::Array(a) => Some(a),
        _ => None,
    }
    .ok_or_else(|| anyhow!("plan must be {{\"plan\": [...]}} or a list"))?;
    if list.is_empty() || list.len() > MAX_ACTIONS {
        bail!("plan must have 1..={MAX_ACTIONS} actions, got {}", list.len());
    }
```

The limits are 6 actions, 120 m altitude, 15 m/s speed and a 500 m range, matching the `sky-schema` defaults. `check_category` then cross-checks the plan against the category: an unsafe order must start with `refuse`, an ambiguous order must start with `clarify`, and a mission order may not use either. Rejected items go to `orders.rejects.jsonl` with their reason, so you can see why the teacher's output was dropped and adjust the prompt.

The two validators split the work. `sky-data` keeps bad training data out of the dataset. `sky-schema::validate` ([chapter 03](./03-typed-decisions)) corrects whatever a model proposes at runtime.

### From orders to decision records

Each valid order also becomes a `choice` record over all 17 action names. The target is the plan's first mission action, skipping a leading `takeoff`, which carries no information:

`crates/sky-data/src/plan.rs`

```rust
pub fn first_mission_action(names: &[String]) -> Option<&str> {
    match names {
        [] => None,
        [only] => Some(only),
        [first, second, ..] if first == "takeoff" => Some(second),
        [first, ..] => Some(first),
    }
}
```

The order text becomes the state and the source tag is `teacher_drone_orders`. About 5% of orders go to `val`, chosen by a hash of the order text. Orders are deduplicated case-insensitively across runs, and the decision file is rebuilt from every order collected so far.

## Commands

```bash
export CARGO_TARGET_DIR=$PWD/target/sky-data
cargo build --release -p sky-data
D=$CARGO_TARGET_DIR/release/sky-data
export SKY_TEACHER_KEY=...        # never commit it; *.env files are gitignored

$D teacher export-questions --max 2000
$D teacher paraphrase-questions --in data/teacher/questions.json --n 3 --max-calls 60
$D teacher drone-orders --n 2000 --batch 40 --max-calls 100
$D teacher --model glm-4.6 --min-interval-ms 3000 drone-orders --n 200
```

`--model`, `--max-calls`, `--batch` and `--min-interval-ms` are global to the `teacher` subcommand. `export-questions` makes no network call and needs no key.

## Status

On the build machine, `paraphrase-questions` wrote 1,999 paraphrase items in 52 calls (`data/teacher/paraphrase.log`), and `drone-orders` holds 3,000 orders after dropping 51 invalid ones in its last run (`data/teacher/orders.log`). Their effect on accuracy was not measured. `scripts/train-ladder.sh` currently trains on `data/decisions,data/games`, so teacher outputs under `data/teacher/` are not part of the default training mix yet. `DATA_LICENSES.md` notes that teacher-generated data is subject to the provider's terms of service.

## Next

[07 A tokenizer with small vocabularies](./07-tokenizer)
