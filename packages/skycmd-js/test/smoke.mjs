// Smoke test of the built package (dist/), runs unchanged on Node >= 18, Deno and Bun:
//   node test/smoke.mjs [model.gguf] [expected.json]
//   deno run --allow-read test/smoke.mjs [model.gguf] [expected.json]
//   bun test/smoke.mjs [model.gguf] [expected.json]
// `expected.json` is an optional `skycmd decide` output for the same request: the WebAssembly
// probabilities must match the native ones.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";

import SkyCmd, { load } from "../dist/skycmd.js";

const modelPath =
  process.argv[2] ?? fileURLToPath(new URL("../../../site/public/models/s1-pico.gguf", import.meta.url));
const expectedPath = process.argv[3];

export const REQUEST = {
  state: "vy 0 dx 36 up 9 dn 7 nxt 10 alt 32",
  questions: {
    action: {
      type: "choice",
      instructions: "Which input keeps the drone flying through the next gap?",
      criteria: {
        thrust: "fire the rotors for an upward impulse",
        glide: "let gravity pull the drone down",
      },
    },
    danger: {
      type: "noul",
      instructions: "Will the drone crash within the next half second if it keeps gliding?",
    },
    margin: {
      type: "score",
      instructions: "How much room is left above and below the drone?",
      criteria: ["none", "tight", "comfortable"],
    },
  },
};

const close = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);
const runtime = globalThis.Deno ? "deno" : globalThis.Bun ? "bun" : `node ${process.version}`;

// path, file: URL and bytes sources give the same model
const model = await SkyCmd.load(modelPath);
const fromUrl = await load(pathToFileURL(modelPath));
const fromBytes = await SkyCmd.load(await readFile(modelPath));

assert.ok(model.nParams > 0);
assert.ok(model.info.hidden_size > 0);
assert.equal(typeof model.name, "string");

const r = model.decide(REQUEST);
assert.deepEqual(Object.keys(r.answers), ["action", "danger", "margin"]);
const { action, danger, margin } = r.answers;
assert.equal(action.type, "choice");
assert.ok(["thrust", "glide"].includes(action.choice));
close(action.probabilities.thrust + action.probabilities.glide, 1, 1e-4, "sum");
assert.ok(danger.noul >= 0 && danger.noul <= 1);
assert.ok(margin.score >= 0 && margin.score <= 2);
assert.equal(margin.legend["2"], "comfortable");
assert.ok(r.usage.input_tokens > 0);

assert.deepEqual(fromUrl.decide(REQUEST), r);
assert.deepEqual(fromBytes.decide(JSON.stringify(REQUEST)), r);
assert.deepEqual(JSON.parse(model.decideJson(JSON.stringify(REQUEST))), r);

// raw path: rendered options in order
const p = model.decideRaw("choice", REQUEST.questions.action.instructions,
  ["thrust: fire the rotors for an upward impulse", "glide: let gravity pull the drone down"],
  REQUEST.state);
assert.equal(p.length, 2);
close(p[0], action.probabilities.thrust, 1e-6, "decideRaw vs decide");
assert.equal(model.scores("noul", "Is it safe?", ["false: no", "true: yes"], "alt 3").length, 2);

// errors
assert.throws(() => model.decide({ questions: {} }), /state/);
assert.throws(() => model.decideRaw("maybe", "?", ["a", "b"], "x"), TypeError);
await assert.rejects(SkyCmd.load(new Uint8Array([1, 2, 3, 4])));
fromUrl.free();
assert.throws(() => fromUrl.decide(REQUEST), /freed/);

// same numbers as the native engine
if (expectedPath) {
  const want = JSON.parse(await readFile(expectedPath, "utf8"));
  for (const [id, a] of Object.entries(want.answers)) {
    const got = r.answers[id];
    for (const [k, v] of Object.entries(a.probabilities ?? {})) {
      close(got.probabilities[k], v, 1e-5, `${id}.${k}`);
    }
    if (a.noul !== undefined) close(got.noul, a.noul, 1e-5, `${id}.noul`);
  }
}

// timing: the raw path in a loop where only the state changes
const t0 = performance.now();
const n = 500;
for (let i = 0; i < n; i++) {
  model.decideRaw("choice", REQUEST.questions.action.instructions, ["thrust", "glide"], `alt ${i % 40} vy 0`);
}
const us = ((performance.now() - t0) * 1000) / n;

console.log(JSON.stringify(r.answers));
console.log(`ok (${runtime}): ${model.name}, ${model.nParams} params, decideRaw ${us.toFixed(1)} us/call`);
