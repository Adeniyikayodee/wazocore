//! atlas-rt vs PyTorch `LlamaForCausalLM` on a tiny random Llama-3
//! (fixtures from `atlas_trace.tiny_reference`, see tests/fixtures/README.md).

use std::fs;

use atlas_rt::forward::{KvCache, Plan, step};
use atlas_rt::ops::{argmax, rope_inv_freq, softmax};
use atlas_rt::weights::weights;
use atlas_satl::format::decode_prefix;
use serde_json::Value;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tiny");

struct Reference {
    tokens: Vec<u32>,
    logits: Vec<Vec<f32>>,
}

fn reference() -> Reference {
    let v: Value = serde_json::from_slice(&fs::read(format!("{DIR}/reference.json")).unwrap()).unwrap();
    let tokens = v["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
    let logits = v["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect())
        .collect();
    Reference { tokens, logits }
}

fn run(satl: &str, tokens: &[u32]) -> Vec<Vec<f32>> {
    let bytes = fs::read(format!("{DIR}/{satl}")).unwrap();
    let (header, data) = decode_prefix(&bytes).unwrap();
    let w = weights(&header, &bytes[data as usize..]).unwrap();
    let plan = Plan::dense(&w.config, rope_inv_freq(&w.config).unwrap());
    let mut kv = KvCache::new(&w.config, tokens.len());
    tokens.iter().map(|&t| step(&w, &plan, &mut kv, t).unwrap()).collect()
}

/// KL(p ‖ q) in nats between softmax distributions of two logit vectors.
fn kl(p_logits: &[f32], q_logits: &[f32]) -> f64 {
    let (p, q) = (softmax(p_logits), softmax(q_logits));
    p.iter().zip(&q).map(|(&a, &b)| if a > 0.0 { a as f64 * (a as f64 / b as f64).ln() } else { 0.0 }).sum()
}

#[test]
fn f32_path_matches_pytorch_at_every_position() {
    let r = reference();
    let ours = run("tiny-f32.satl", &r.tokens);
    for (pos, (a, b)) in ours.iter().zip(&r.logits).enumerate() {
        let max_diff = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "position {pos}: max |Δlogit| = {max_diff}");
        assert_eq!(argmax(a), argmax(b), "position {pos}: greedy token differs");
    }
}

#[test]
fn q4_path_stays_close_to_pytorch() {
    let r = reference();
    let ours = run("tiny-q4.satl", &r.tokens);
    let kls: Vec<f64> = ours.iter().zip(&r.logits).map(|(a, b)| kl(b, a)).collect();
    let mean_kl = kls.iter().sum::<f64>() / kls.len() as f64;
    let agree = ours.iter().zip(&r.logits).filter(|(a, b)| argmax(a) == argmax(b)).count();
    eprintln!("q4 vs torch: mean KL {mean_kl:.5} nats, top-1 agreement {agree}/{}", r.tokens.len());
    assert!(mean_kl < 0.05, "mean KL {mean_kl}");
}
