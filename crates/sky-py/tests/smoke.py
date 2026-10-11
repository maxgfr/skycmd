"""Smoke test of the built wheel.

    uv run --with target/wheels/skycmd-*.whl python crates/sky-py/tests/smoke.py site/public/models/s1-pico.gguf
"""

import json
import math
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import skycmd

REQUEST = {
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
        "margin": {
            "type": "score",
            "instructions": "How much room is left above and below the drone?",
            "criteria": ["none", "tight", "comfortable"],
        },
    },
}


def main(path: str) -> None:
    m = skycmd.Model(path)
    assert isinstance(skycmd.__version__, str)
    assert m.n_params > 0 and m.path == path, m
    assert m.info["hidden_size"] > 0
    print(repr(m))

    r = m.decide(REQUEST)
    assert list(r["answers"]) == ["action", "danger", "margin"], r
    action = r["answers"]["action"]
    assert action["choice"] in ("thrust", "glide")
    assert math.isclose(sum(action["probabilities"].values()), 1.0, abs_tol=1e-4)
    assert 0.0 <= r["answers"]["danger"]["noul"] <= 1.0
    assert 0.0 <= r["answers"]["margin"]["score"] <= 2.0
    assert r["usage"]["input_tokens"] > 0

    # JSON text in, same answer; JSON text out
    assert m.decide(json.dumps(REQUEST)) == r
    assert json.loads(m.decide_json(json.dumps(REQUEST))) == r

    # raw path: rendered options, probabilities in order
    p = m.decide_raw(
        "choice",
        REQUEST["questions"]["action"]["instructions"],
        ["thrust: fire the rotors for an upward impulse", "glide: let gravity pull the drone down"],
        REQUEST["state"],
    )
    assert len(p) == 2 and math.isclose(sum(p), 1.0, abs_tol=1e-4)
    assert math.isclose(p[0], action["probabilities"]["thrust"], abs_tol=1e-5), (p, action)

    # bytes and os.PathLike sources
    data = pathlib.Path(path).read_bytes()
    assert skycmd.Model(data).decide(REQUEST)["answers"] == r["answers"]
    assert skycmd.load(pathlib.Path(path)).path == path

    # threads share one model
    with ThreadPoolExecutor(4) as pool:
        outs = list(pool.map(lambda _: m.decide(REQUEST), range(16)))
    assert all(o == r for o in outs)

    # errors
    for bad, exc in [
        ({"questions": {}}, ValueError),
        ({"state": "x", "questions": {"q": {"type": "maybe", "instructions": "?"}}}, ValueError),
    ]:
        try:
            m.decide(bad)
        except exc:
            pass
        else:
            raise AssertionError(f"no {exc.__name__} for {bad}")
    try:
        m.decide_raw("maybe", "?", ["a", "b"], "x")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown question type accepted")
    try:
        skycmd.Model(b"not a gguf")
    except ValueError:
        pass
    else:
        raise AssertionError("garbage bytes accepted")
    try:
        skycmd.Model("/nonexistent/model.gguf")
    except OSError:
        pass
    else:
        raise AssertionError("missing file accepted")

    print(json.dumps(r["answers"], indent=2))
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "site/public/models/s1-pico.gguf")
