//! End-to-end smoke test of the public `Model` API on native.
//!
//! Mirrors the JS-side flow: load bytes → encode prompt → loop step → decode token IDs.
//!
//! Build:
//!   cargo run --release --example model_api -- <gguf> [prompt] [n_predict]

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use rullama::api::Model;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => { eprintln!("usage: model_api <gguf> [prompt] [n_predict]"); return ExitCode::from(2); }
    };
    let prompt = args.next().unwrap_or_else(|| "Hello, world!".to_string());
    let n_predict: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("loading model ...");
    let t0 = Instant::now();
    let bytes = fs::read(&path).expect("read");
    let mut model = pollster::block_on(Model::load_native(bytes)).expect("load");
    println!("  loaded in {:?}", t0.elapsed());
    println!("  vocab_size = {}, position = {}", model.vocab_size_native(), model.position_native());

    let prompt_ids = model.encode_tokens(&prompt);
    println!("prompt = {prompt:?}");
    println!("prompt_ids = {prompt_ids:?}");

    let mut next: u32 = 0;
    let t0 = Instant::now();
    for &id in &prompt_ids {
        next = pollster::block_on(model.step_native(id)).expect("step");
    }
    println!("after prompt: pos={}, predicted_next = {} ({:?})",
        model.position_native(), next, model.token_str_native(next).unwrap_or_default());

    print!("generation:");
    let mut emitted = vec![next];
    for _ in 0..n_predict.saturating_sub(1) {
        if model.is_eos_native(next) { break; }
        let token = next;
        next = pollster::block_on(model.step_native(token)).expect("step gen");
        emitted.push(next);
    }
    for id in &emitted {
        print!(" {} ({:?})", id, model.token_str_native(*id).unwrap_or_default());
    }
    println!();
    println!("total time: {:?}", t0.elapsed());

    ExitCode::SUCCESS
}
