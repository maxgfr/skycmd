//! Shared data contracts of skycmd.
//!
//! - [`record`]: the JSONL decision record consumed by the trainer, plus the option renderers
//!   that reproduce `render_options` of the Laya reference runtime.
//! - [`systemone`]: request/response types of the llama.cpp `/v1/systemone` endpoint.
//! - [`plan`]: the drone command-plan schema and its safety validator.

pub mod plan;
pub mod record;
pub mod systemone;

pub use plan::{validate, Action, Plan, SafetyLimits, Telemetry, Verdict};
pub use record::{
    render_choice, render_noul, render_score, DecisionRecord, QType, Split, NOUL_FALSE_DEFAULT,
    NOUL_TRUE_DEFAULT,
};
pub use systemone::{Answer, NoulCriteria, Question, SystemOneRequest, SystemOneResponse};
