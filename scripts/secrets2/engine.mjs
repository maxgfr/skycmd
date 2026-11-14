#!/usr/bin/env node
// s6-secrets v2 (docs/s6-secrets2-prereg.md, sections 6 and 7.7): secretgate's engine from a
// checkout, on a list of files. Started by run_secretgate.py and clean_text.py, which build
// the list after the split guard and check that the checkout is the frozen one; this script
// checks every path again against the repositories it is given.
//
//   node engine.mjs --secretgate <checkout> --mode default|candidates --files-from LIST
//        --base DIR --out FILE [--repos id,id,...] [--no-path] [--no-pragmas] [--stats FILE]
//        [--where FILE]
//
// --mode default      the default engine, filters on: `scan(text, {sourcePath})` per file
//                     (section 6). With --no-path the engine gets no source path, with
//                     --no-pragmas it ignores inline pragmas: how the hooks scan tool output,
//                     and how the census of the clean-text corpus was run (section 7.7).
//                     One line per finding: file, rule, line, endLine, cpStart, cpEnd. No value.
// --mode candidates   the generator sg-cand-1 through `candidateRecords` of the checkout's
//                     scripts/rule-candidates.mjs, WITHOUT a source path (so the global path
//                     allowlist does not apply); the file's path and extension are written to
//                     the record afterwards. Only for the clean-text corpus read as the census
//                     read it; with a path the generator is the checkout's own command line.
//                     The records hold RAW values: the output stays in the data folder.
//                     With --where FILE the records carry no path at all (file null, no
//                     extension: an issue text has none, section 7.4) and FILE gets, line for
//                     line, the path each record comes from.
//
// Lines are those of the engine ("\n" ends a line); cpStart and cpEnd are columns in Unicode
// code points, cpEnd on endLine. A file that is empty, larger than --max-bytes or has a NUL
// byte in its first 8192 bytes is skipped, as the generator skips it.
import { closeSync, fchmodSync, openSync, readFileSync, statSync, writeFileSync, writeSync } from "node:fs";
import * as nodeModule from "node:module";
import { extname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const opt = { maxBytes: 64 * 1024 * 1024, path: true, pragmas: true };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === "--secretgate") opt.secretgate = resolve(argv[++i]);
  else if (a === "--mode") opt.mode = argv[++i];
  else if (a === "--files-from") opt.filesFrom = resolve(argv[++i]);
  else if (a === "--base") opt.base = resolve(argv[++i]);
  else if (a === "--out") opt.out = resolve(argv[++i]);
  else if (a === "--stats") opt.stats = resolve(argv[++i]);
  else if (a === "--where") opt.where = resolve(argv[++i]);
  else if (a === "--repos") opt.repos = new Set(argv[++i].split(","));
  else if (a === "--max-bytes") opt.maxBytes = Number(argv[++i]);
  else if (a === "--no-path") opt.path = false;
  else if (a === "--no-pragmas") opt.pragmas = false;
  else throw new Error(`engine.mjs: unknown argument ${a}`);
}
if (!opt.secretgate || !opt.filesFrom || !opt.base || !opt.out || !["default", "candidates"].includes(opt.mode)) {
  throw new Error("usage: engine.mjs --secretgate <checkout> --mode default|candidates --files-from LIST --base DIR --out FILE [--repos ids] [--no-path] [--no-pragmas] [--stats FILE]");
}

const files = readFileSync(opt.filesFrom, "utf8")
  .split("\n")
  .filter((l) => l !== "");
// The guard: with --repos, every path is a file of one of these CredData repositories.
if (opt.repos) {
  for (const f of files) {
    const m = /^data\/([0-9a-f]{8})\//.exec(f);
    if (!m || !opt.repos.has(m[1])) throw new Error(`refused: ${f} is not a file of the repositories this run was given`);
  }
}

nodeModule.registerHooks({
  resolve(spec, ctx, next) {
    try {
      return next(spec, ctx);
    } catch (err) {
      if (spec.startsWith(".") && spec.endsWith(".js")) return next(`${spec.slice(0, -3)}.ts`, ctx);
      throw err;
    }
  },
});
const engine = await import(pathToFileURL(join(opt.secretgate, "src/engine/scanner.ts")).href);
const candidates = opt.mode === "candidates" ? await import(pathToFileURL(join(opt.secretgate, "scripts/rule-candidates.mjs")).href) : undefined;

function readText(file) {
  const st = statSync(file, { throwIfNoEntry: false });
  if (st === undefined || !st.isFile() || st.size === 0 || st.size > opt.maxBytes) return undefined;
  const buf = readFileSync(file);
  if (buf.subarray(0, 8192).includes(0)) return undefined;
  return buf.toString("utf8");
}

function lineStartsOf(text) {
  const starts = [0];
  for (let i = text.indexOf("\n"); i !== -1; i = text.indexOf("\n", i + 1)) starts.push(i + 1);
  return starts;
}

function lineIndexAt(starts, offset) {
  let lo = 0;
  let hi = starts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (starts[mid] <= offset) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

// How many Unicode code points `text` holds between two offsets.
function codePoints(text, from, to) {
  let n = to - from;
  for (let i = from; i < to - 1; i++) {
    const c = text.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff) {
      const d = text.charCodeAt(i + 1);
      if (d >= 0xdc00 && d <= 0xdfff) {
        n--;
        i++;
      }
    }
  }
  return n;
}

const began = performance.now();
const cpu0 = process.cpuUsage();
const fd = openSync(opt.out, "w", 0o600);
fchmodSync(fd, 0o600);
const whereFd = opt.where ? openSync(opt.where, "w", 0o600) : undefined;
const stats = { mode: opt.mode, sourcePath: opt.mode === "default" ? opt.path : false, pragmas: opt.pragmas, files: files.length, read: 0, skipped: 0, filesWithFindings: 0, findings: 0, byRule: {} };
for (const file of files) {
  const text = readText(join(opt.base, file));
  if (text === undefined) {
    stats.skipped++;
    continue;
  }
  stats.read++;
  let n = 0;
  if (opt.mode === "default") {
    const cfg = {};
    if (opt.path) cfg.sourcePath = file;
    if (!opt.pragmas) cfg.pragmas = false;
    const starts = lineStartsOf(text);
    for (const f of engine.scan(text, cfg)) {
      const line = lineIndexAt(starts, f.start);
      const endLine = lineIndexAt(starts, Math.max(f.start, f.end - 1));
      writeSync(fd, `${JSON.stringify({ file, rule: f.ruleId, line: line + 1, endLine: endLine + 1, cpStart: codePoints(text, starts[line], f.start), cpEnd: codePoints(text, starts[endLine], f.end) })}\n`);
      stats.byRule[f.ruleId] = (stats.byRule[f.ruleId] ?? 0) + 1;
      n++;
    }
  } else {
    for (const c of candidates.candidateRecords(text, engine, { raw: true, maxLine: 1000 })) {
      if (whereFd === undefined) writeSync(fd, `${JSON.stringify({ ...c, file, ext: extname(file).toLowerCase() })}\n`);
      else {
        writeSync(fd, `${JSON.stringify(c)}\n`);
        writeSync(whereFd, `${JSON.stringify({ file })}\n`);
      }
      stats.byRule[c.rule] = (stats.byRule[c.rule] ?? 0) + 1;
      n++;
    }
  }
  stats.findings += n;
  if (n > 0) stats.filesWithFindings++;
}
closeSync(fd);
if (whereFd !== undefined) closeSync(whereFd);
const cpu = process.cpuUsage(cpu0);
stats.seconds = Number(((performance.now() - began) / 1000).toFixed(1));
stats.cpuSeconds = Number(((cpu.user + cpu.system) / 1e6).toFixed(1));
if (opt.stats) writeFileSync(opt.stats, `${JSON.stringify(stats, null, 2)}\n`);
console.error(`engine.mjs: ${opt.mode}${opt.path ? "" : " (no source path)"}: ${stats.read} files, ${stats.findings} findings in ${stats.filesWithFindings} files, ${stats.seconds} s`);
