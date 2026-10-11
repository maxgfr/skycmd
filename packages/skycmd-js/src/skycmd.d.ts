// Types of the `skycmd` package: the System One request / response (`POST /v1/systemone`) and
// the model wrapper.

/** Pick one option among named options. */
export interface ChoiceQuestion {
  type: "choice";
  instructions: string;
  /** Option key -> description (an empty description renders the key alone). */
  criteria: Record<string, string>;
}

/** Rate on an ordinal scale of 2 to 10 levels (level i = index i). */
export interface ScoreQuestion {
  type: "score";
  instructions: string;
  criteria: string[];
}

/** Yes / no: the answer is p(true). */
export interface NoulQuestion {
  type: "noul";
  instructions: string;
  /** Optional descriptions of the two outcomes. */
  criteria?: { false?: string; true?: string };
}

export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;
export type QuestionType = Question["type"];

/** The body of `POST /v1/systemone`. */
export interface DecisionRequest {
  /** Free text or compact game state; non-string JSON is serialized. */
  state: unknown;
  /** Question id -> question. Answers keep this order. */
  questions: Record<string, Question>;
}

export interface ChoiceAnswer {
  type: "choice";
  /** Key of the most likely option. */
  choice: string;
  probabilities: Record<string, number>;
  /** (pmax - 1/n) / (1 - 1/n). */
  confidence: number;
}

export interface ScoreAnswer {
  type: "score";
  /** Expected level: sum of i * p_i. */
  score: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
  confidence: number;
}

export interface NoulAnswer {
  type: "noul";
  /** Probability that the statement holds. */
  noul: number;
}

export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export interface DecisionResponse {
  model: string;
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number };
  /** s2 models: index of the early exit taken and encoder layers used (per answer as well). */
  [extra: string]: unknown;
}

export interface ModelInfo {
  name: string;
  n_params: number;
  vocab_size: number;
  hidden_size: number;
  num_layers: number;
  num_heads: number;
  intermediate_size: number;
  head_layers: number;
  max_len: number;
  head_max_len: number;
  q8: boolean;
  temperatures: Record<string, number>;
  [extra: string]: unknown;
}

/** Anything a model (or the engine's .wasm file) can be loaded from. */
export type Source = string | URL | Response | Blob | ArrayBuffer | ArrayBufferView;

export interface LoadOptions {
  /** Compute with int8 weights (smaller memory, slightly different probabilities). */
  q8?: boolean;
  /** With `q8`: single-pass activation quantization (faster, less accurate). */
  fast?: boolean;
  /** Where the engine's `sky_infer_bg.wasm` comes from (default: next to skycmd.js). */
  wasm?: Source | WebAssembly.Module;
}

/** A loaded decision model (GGUF: s1 `modern-bert` or s2 `skycmd-s2`). */
export declare class SkyCmd {
  private constructor();
  /** Load a model from a URL, a path (Node, Deno, Bun), a Response, a Blob or bytes. */
  static load(source: Source, options?: LoadOptions): Promise<SkyCmd>;
  /** Answer a System One request (object or JSON text), like `POST /v1/systemone`. */
  decide(request: DecisionRequest | string): DecisionResponse;
  /** Same as `decide` with JSON text in and out. */
  decideJson(request: string): string;
  /**
   * Probabilities (after the temperature) of already rendered option texts, in order, e.g.
   * `["thrust: fire the rotors", "glide"]` or, for noul, `["false: ...", "true: ..."]`.
   */
  decideRaw(type: QuestionType, instructions: string, options: string[], state: string): number[];
  /** Raw scores (before the temperature). */
  scores(type: QuestionType, instructions: string, options: string[], state: string): number[];
  readonly info: ModelInfo;
  readonly name: string;
  readonly nParams: number;
  /** Free the WebAssembly memory of this model. */
  free(): void;
}

/** Load the WebAssembly engine once (called by `SkyCmd.load`). */
export declare function init(wasm?: Source | WebAssembly.Module): Promise<void>;
/** Same as `SkyCmd.load`. */
export declare function load(source: Source, options?: LoadOptions): Promise<SkyCmd>;
export default SkyCmd;
