# skycmd models in LM Studio

skycmd `s1` GGUFs use the llama.cpp architecture `modern-bert` with a `laya` decision head
(`modern-bert.decision.*` keys). They are not chat or embedding models: they score the options
of typed questions (`choice`, `score`, `noul`) in one forward pass and are queried through the
System One API, `POST /v1/systemone`.

## Status

LM Studio's local server exposes OpenAI-compatible endpoints (`/v1/chat/completions`,
`/v1/completions`, `/v1/embeddings`, `/v1/models`) but no `/v1/systemone`, so it cannot answer
decision requests even when its llama.cpp runtime knows the architecture. The import and load
steps below are therefore only useful to keep the files in your LM Studio library; see
[docs/compat.md](../../docs/compat.md) for what was tested.

`s2` GGUFs (`skycmd-s2`) are served only by the skycmd engine.

## Import an s1 model

```sh
curl -LO https://maxgfr.github.io/skycmd/models/s1-pico.gguf
lms import --copy --user-repo skycmd/s1-pico s1-pico.gguf
lms ls
```

## Serve decisions instead

The same file runs as a System One server with the `skycmd` CLI or with llama.cpp:

```sh
skycmd serve --model s1-pico.gguf --port 8080
# or: llama-server -m s1-pico.gguf -ngl 0 --port 8080

curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "vy 0 dx 36 up 9 dn 7 nxt 10 alt 32",
  "questions": {
    "action": {
      "type": "choice",
      "instructions": "Which input keeps the drone flying through the next gap?",
      "criteria": {"thrust": "fire the rotors for an upward impulse", "glide": "let gravity pull the drone down"}
    }
  }
}'
```

Install guide: <https://maxgfr.github.io/skycmd/install/>.
