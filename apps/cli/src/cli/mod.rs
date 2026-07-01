pub mod analytics;
pub mod app;
pub mod attach;
pub mod auth;
pub mod chat;
pub mod config;
pub mod cost;
pub mod first_run;
pub mod history;
pub mod local_models;
pub mod local_models_setup;
pub mod mcp;
pub mod models;
pub mod plan;
// LEGACY: Studio remote-bridge (web access to CLI agents) — discontinued,
// disabled by default, likely to return.
#[cfg(feature = "remote-bridge")]
pub mod remote;
pub mod self_improve_cmd;
pub mod task;
