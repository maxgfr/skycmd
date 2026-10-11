# 16 Bonus: llama.cpp /v1/systemone and wllama

The GGUF files from chapter 13 use the same keys and tensor names as llama.cpp's own converter, so you are not tied to our engine. This chapter shows how to serve a converted model with llama.cpp's decision server (`POST /v1/systemone`) and how to run it in the browser with wllama. It is careful about which parts are upstream, which come from a pull request, and which we have not tested.

## What is upstream, what is not, what is untested

| piece | status | source |
|---|---|---|
| `modern-bert` decision head in llama.cpp (`build_decision_head`) | in llama.cpp PR #29818, which docs/research.md records as merged on 2026-10-02 | `models-src/modern-bert.cpp` |
| `/v1/systemone` server (`server_decision_context`) | same PR | `models-src/server-decision.cpp`, `.h` |
| HF → GGUF converter for Laya checkpoints (`ModernBertDecisionModel`) | same PR | `models-src/bert.py` |
| wllama 3.8.1 with `createSystemOne()` | listed as a runtime in docs/research.md | not in this repo |
| skycmd GGUF loaded by `llama-server` | **not tested in this repo** | — |
| skycmd GGUF loaded by wllama | **not tested in this repo** | — |

The files under `models-src/` are local, gitignored, read-only copies that we used as the specification. Whether a given llama.cpp release contains exactly this code is something you should check against the release you install. Everything this repo verifies runs through `sky-infer`: the golden tests against the Python Laya runtime (chapter 13), and our own reader of our own GGUF files.

## One file, two readers

`sky-convert` follows `bert.py` key by key, which you can see by comparing the two:

| what | `bert.py` (llama.cpp) | `sky-convert` (`convert.rs`) |
|---|---|---|
| architecture | `gguf.MODEL_ARCH.MODERN_BERT` | `general.architecture = "modern-bert"` |
| decision type | `add_decision_type(gguf.DecisionType.LAYA)` | `modern-bert.decision.type = "laya"` |
| head depth | `add_decision_block_count(head_layers)` | `modern-bert.decision.block_count` |
| head budget | `add_decision_max_head_tokens(head_max_len)` | `modern-bert.decision.max_head_tokens` |
| temperatures | `choice:3-5` → `choice.3_5` | same rewrite (`laya_bucket_to_key`) |
| head blocks | renumbered after the encoder blocks | `blk.{n_enc + i}.*` |
| type embedding | `token_types.weight`, `token_type_count = 3` | same |
| prompt | `systemone` Jinja chat template | same template (`systemone_template`) |

`sky-train export` writes the directory layout `bert.py` detects (`encoder/config.json` plus `rl_agent_config.json`, with the tokenizer in `tokenizer/`). In principle llama.cpp's `convert_hf_to_gguf.py` can therefore convert `models/s1-pico` too. We have not tried it, and there is one likely obstacle. llama.cpp's converter recognizes BPE pre-tokenizers by a checksum of known tokenizers, and a freshly trained 1k-token vocabulary is probably not among them. `sky-convert` avoids the question: it writes `tokenizer.ggml.pre = "gpt-2"` directly (override with `--pre NAME`).

## Serving with `llama-server`

The commands below are what we would run. They have **not** been executed against a skycmd model in this repo.

```sh
# 1. a GGUF file (the site already ships s1-pico)
ls -l site/public/models/s1-pico.gguf

# 2. llama.cpp built from a revision that includes PR #29818
llama-server -m site/public/models/s1-pico.gguf -c 256 --port 8080

# 3. one request
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @request.json
```

`-c 256` matches the Laya `max_len` of our ladder. `convert.rs` notes that llama.cpp does not read `decision.max_tokens` and limits the prompt only to the context size. With a larger context, prompts longer than the model was trained on would be accepted rather than truncated.

When the server loads a Laya model, `server_decision_context::init` checks for the mask and separator tokens and a non-zero `max_head_tokens`.

From `models-src/server-decision.cpp`:

```cpp
    } else if (model_type == COMMON_DECISION_TYPE_LAYA) {
        token_marker = llama_vocab_mask(vocab);
        token_sep    = llama_vocab_sep(vocab);
        if (token_marker == LLAMA_TOKEN_NULL || token_sep == LLAMA_TOKEN_NULL) {
            throw std::runtime_error("decision model has no mask or sep token");
        }
        text_marker = common_token_to_piece(vocab, token_marker, true);

        const std::string val = decision_meta_str(model, prefix + "max_head_tokens");
        max_head_tokens = std::strtoul(val.c_str(), nullptr, 10);
        if (max_head_tokens == 0) {
            throw std::runtime_error("decision model has no valid max_head_tokens");
        }
        n_options_max = 255;
```

Our files carry `tokenizer.ggml.mask_token_id = 4`, `seperator_token_id = 3` (llama.cpp's spelling) and `decision.max_head_tokens = 96`, so these checks should pass.

## The request

The shape comes from `parse_questions` in `server-decision.cpp`, and `crates/sky-schema/src/systemone.rs` mirrors it as Rust types (`SystemOneRequest`, `Question`, `Answer`). A Flappy Drone request, using the same questions as the game:

```json
{
  "state": "vy 0 dx 36 up 9 dn 7 nxt 10 alt 32",
  "questions": {
    "action": {
      "type": "choice",
      "instructions": "Which input keeps the drone flying through the next gap?",
      "criteria": {
        "thrust": "fire the rotors for an upward impulse",
        "glide": "let gravity pull the drone down"
      }
    },
    "danger": {
      "type": "noul",
      "instructions": "Will the drone crash within the next half second if it keeps gliding?"
    },
    "margin": {
      "type": "score",
      "instructions": "How much room is left above and below the drone?",
      "criteria": ["none", "tight", "comfortable"]
    }
  }
}
```

`margin` is an extra question that shows the score type; the game itself does not ask it, and s1-pico was not trained on it. The validation rules, in the server's own words:

From `models-src/server-decision.cpp`:

```cpp
        if (type_name == "choice") {
            question.type = SERVER_DECISION_QUESTION_CHOICE;
            if (!criteria.is_object() || criteria.empty()) {
                throw err("\"criteria\" must be a non-empty object");
            }
            for (const auto & [key, description] : criteria.items()) {
                question.options.push_back({key, description});
            }
            if (choice_sorted) {
                std::sort(question.options.begin(), question.options.end(), [](const auto & a, const auto & b) {
                    return a.key < b.key;
                });
            }
        } else if (type_name == "score") {
            question.type = SERVER_DECISION_QUESTION_SCORE;
            if (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10) {
                throw err("\"criteria\" must be an array of 2 to 10 levels");
            }
```

- `state` is required and may be any JSON value. Non-string values are serialized by the template (`tojson`).
- `questions` is a non-empty object; answers come back under the same ids, in order.
- `choice` needs an object of `key → description`, where a description may be `null`. `choice_sorted` is only set for the Clef model type, so Laya keeps your key order.
- `score` needs 2 to 10 levels; level `i` is rendered `level i: …`.
- `noul` accepts an optional `{"true": …, "false": …}` object; the defaults are `yes, the statement holds` / `no, the statement does not hold`.

## How the server builds the prompt

The server renders the `systemone` template stored in the GGUF, tokenizes it, then cuts the question and options to `max_head_tokens` with the same rule as training (`fill_task_laya`):

From `models-src/server-decision.cpp`:

```cpp
    set_max(max_option_tokens + 1);
    if (n_options_tokens + 16 > max_head_tokens) {
        // too many or too long options, shrink them evenly
        set_max(std::max((size_t) 4, (max_head_tokens - std::min(max_head_tokens, (size_t) 16)) / n_options));
    }
    const size_t n_question_max = std::max((size_t) 8, max_head_tokens - std::min(max_head_tokens, n_options_tokens));
```

`sky-infer`'s `prompt::build_prefix` implements the same three steps. Options are capped at 48 tokens plus their marker; if they leave fewer than 16 tokens, all of them shrink evenly, with a floor of 4. The question then gets what is left, at least 8 tokens. The golden tests require `sky-infer`'s token ids to match the Python Laya runtime exactly; the server's rule is the same on paper, but we have not compared its token ids.

The server then reads the score column of the question's type at each marker, applies `softmax(score / T)` with the bucketed temperature, and formats the answer.

## The response

Each answer comes from `format_answer`.

From `models-src/server-decision.cpp`:

```cpp
    if (question.type == SERVER_DECISION_QUESTION_CHOICE) {
        const size_t best = std::max_element(probs.begin(), probs.end()) - probs.begin();
        answer["choice"]        = question.options[best].key;
        answer["probabilities"] = probabilities;
        answer["confidence"]    = decision_confidence_choice(probs);
    } else {
        double expected = 0.0;
        json legend = json::object();
        for (size_t i = 0; i < n; i++) {
            expected += i * probs[i];
            legend[question.options[i].key] = question.options[i].description;
        }
        answer["score"]         = expected;
        answer["legend"]        = legend;
        answer["probabilities"] = probabilities;
        answer["confidence"]    = decision_confidence_score(probs);
    }
```

A noul answer is just `{"type": "noul", "noul": p_true}`. The envelope around the answers is assembled outside `server-decision.cpp`, in a file we do not have, so we cannot quote it. `SystemOneResponse` in `sky-schema` therefore reads both `{"answers": {...} }` (ignoring extra top-level fields) and a bare `{"<id>": answer}` map. The answers look like this (from the `sky-schema` unit test, with illustrative probabilities):

```json
{"answers":{
  "action":{"type":"choice","choice":"thrust","probabilities":{"thrust":0.75,"glide":0.25},"confidence":0.5},
  "risk":{"type":"score","score":0.75,"legend":{"0":"negligible","1":"low","2":"critical"},
          "probabilities":{"0":0.5,"1":0.25,"2":0.25},"confidence":0.0},
  "danger":{"type":"noul","noul":0.125}}}
```

`sky-infer`'s `decide_json` writes `{"model": …, "answers": {…}, "usage": {"input_tokens": n, "output_tokens": 0} }`, plus `"micros"` when built with the `timing` feature. You can answer the same `request.json` locally, without llama.cpp:

```sh
export CARGO_TARGET_DIR=$PWD/target/sky-infer
cargo run --release -p sky-infer --example decide -- site/public/models/s1-pico.gguf request.json
```

## Known differences between `sky-infer` and `llama-server`

| behavior | `sky-infer` | `llama-server` (PR #29818 copy) |
|---|---|---|
| `choice` criteria as a JSON array | accepted (Laya runtime behavior: keys without descriptions) | rejected: must be an object |
| `score` level count | any non-empty array | 2 to 10 |
| head evaluation | only the question's type, last head layer only at markers | all three types, scores concatenated, one column read |
| sequence limit | `decision.max_tokens` (256) | context size (`-c`) |
| `skycmd.tokenizer.*` keys (NFC normalizer, lstrip/rstrip) | applied | ignored |
| tokenizer equality with HF `tokenizers` | tested (`tests/tokenizer.rs`) | not tested for our vocabularies |

Probability agreement between the two runtimes on skycmd models: TBD. To check it, send the same `request.json` to both and compare the `probabilities` fields. On F16 files, a difference much larger than the 1e-3 our golden tests allow would point to a tokenization or template difference.

## In the browser with wllama

wllama is a WebAssembly build of llama.cpp for browsers. docs/research.md lists wllama 3.8.1 as a runtime for decision models, with a `createSystemOne()` entry point. We have not used it in this repo, and we have not checked the exact signature of `createSystemOne()` or the shape of what it returns. Read the wllama README for the version you install. A sketch, **unverified**:

```js
// Not tested in this repo: check the wllama 3.8.x README for the real API.
import { Wllama } from "@wllama/wllama";

const wllama = new Wllama(/* paths to the wllama .wasm files */);
await wllama.loadModelFromUrl("/skycmd/models/s1-pico.gguf");
// createSystemOne() is listed in docs/research.md; its arguments are not verified here.
```

How it compares with our `sky-web` module, as far as we know:

| | `sky-web` (this project) | wllama |
|---|---|---|
| what runs in WASM | engine and game cores, one frame per call | llama.cpp (any supported GGUF) |
| model support | Laya / ModernBERT decision models only | everything llama.cpp supports |
| SIMD | SIMD128, f32 compute | TBD |
| threads | single-threaded, main thread | multi-threading needs cross-origin isolation (COOP/COEP headers); GitHub Pages cannot set headers |
| module size | about 0.6 MB including the games | TBD |
| latency per decision for s1-pico | about 640 µs (2 questions + physics, Node 24) | TBD |

Use wllama when you want one browser runtime for many model families, or the exact llama.cpp numerics. Use `sky-web` when the model must step a game at 60 frames per second next to the physics, which is what this site does.

## Next

[17 Scaling up and troubleshooting](./17-scaling-and-troubleshooting)
