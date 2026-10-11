# skycmd (Python)

Python bindings of the [skycmd](https://github.com/maxgfr/skycmd) engine. skycmd models are tiny
"System One" decision models: a bidirectional encoder with a typed decision head that scores
options in one forward pass, without generating any text. The bindings load `s1` (`modern-bert`)
and `s2` (`skycmd-s2`) GGUF files.

```sh
pip install skycmd            # or: uv add skycmd
```

```python
import skycmd

m = skycmd.Model("s1-pico.gguf")          # a path, os.PathLike, bytes or bytearray
r = m.decide({
    "state": "vy 0 dx 36 up 9 dn 7 nxt 10 alt 32",
    "questions": {
        "action": {
            "type": "choice",
            "instructions": "Which input keeps the drone flying through the next gap?",
            "criteria": {
                "thrust": "fire the rotors for an upward impulse",
                "glide": "let gravity pull the drone down",
            },
        },
        "danger": {
            "type": "noul",
            "instructions": "Will the drone crash within the next half second if it keeps gliding?",
        },
    },
})
r["answers"]["action"]["choice"]          # "thrust" or "glide"
r["answers"]["danger"]["noul"]            # probability of "yes"

# Already rendered options, fastest path for a loop where only the state changes:
m.decide_raw("choice", "Which input keeps the drone flying?",
             ["thrust: fire the rotors", "glide: let gravity pull"], "vy 0 dx 36 alt 32")
```

`decide` takes a dict or JSON text with the body of `POST /v1/systemone` (the llama.cpp System One
API) and returns the response as a dict. `decide_json` takes and returns JSON text. Calls release
the GIL and a `Model` can be shared between threads.

Models: <https://maxgfr.github.io/skycmd/install/>. License: Apache-2.0.
