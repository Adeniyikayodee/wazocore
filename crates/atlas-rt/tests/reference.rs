//! atlas-rt vs PyTorch `LlamaForCausalLM` on a tiny random Llama-3
//! (fixtures from `atlas_trace.tiny_reference`, see tests/fixtures/README.md).

use std::fs;

use atlas_rt::forward::{Kernels, KvCache, Plan, Want, prefill, step};
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
    run_with(satl, tokens, Kernels::Reference)
}

fn run_with(satl: &str, tokens: &[u32], kernels: Kernels) -> Vec<Vec<f32>> {
    let bytes = fs::read(format!("{DIR}/{satl}")).unwrap();
    let (header, data) = decode_prefix(&bytes).unwrap();
    let w = weights(&header, &bytes[data as usize..]).unwrap();
    let plan = Plan::dense(&w.config, rope_inv_freq(&w.config).unwrap(), kernels);
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

/// Int8 activations change logits slightly; the fast path must stay far
/// closer to the f32 reference than q4 weights are to PyTorch (0.038 nats).
#[test]
fn fast_kernels_stay_close_to_reference_end_to_end() {
    let r = reference();
    let (slow, quick) = (run("tiny-q4.satl", &r.tokens), run_with("tiny-q4.satl", &r.tokens, Kernels::Fast));
    let kls: Vec<f64> = quick.iter().zip(&slow).map(|(a, b)| kl(b, a)).collect();
    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
    let agree = quick.iter().zip(&slow).filter(|(a, b)| argmax(a) == argmax(b)).count();
    eprintln!("fast vs reference: mean KL {mean:.2e} nats, top-1 {agree}/{}", r.tokens.len());
    assert!(mean < 1e-3, "mean KL {mean}");
}

/// Batched prefill must equal token-by-token decode exactly, including
/// across batch boundaries and when decoding continues afterwards.
#[test]
fn prefill_equals_stepwise_decode_bit_for_bit() {
    let r = reference();
    let bytes = fs::read(format!("{DIR}/tiny-q4.satl")).unwrap();
    let (header, data) = decode_prefix(&bytes).unwrap();
    let w = weights(&header, &bytes[data as usize..]).unwrap();
    let plan = Plan::dense(&w.config, rope_inv_freq(&w.config).unwrap(), Kernels::Fast);
    let n = r.tokens.len();

    let mut kv_a = KvCache::new(&w.config, n);
    let stepwise: Vec<Vec<f32>> = r.tokens.iter().map(|&t| step(&w, &plan, &mut kv_a, t).unwrap()).collect();

    let mut kv_b = KvCache::new(&w.config, n);
    let mut batched = prefill(&w, &plan, &mut kv_b, &r.tokens[..25], Want::All).unwrap();
    batched.extend(prefill(&w, &plan, &mut kv_b, &r.tokens[25..30], Want::All).unwrap());
    batched.extend(r.tokens[30..].iter().map(|&t| step(&w, &plan, &mut kv_b, t).unwrap()));
    assert_eq!(batched, stepwise);
    assert_eq!(kv_a, kv_b);

    let mut kv_c = KvCache::new(&w.config, n);
    let last = prefill(&w, &plan, &mut kv_c, &r.tokens, Want::Last).unwrap();
    assert_eq!(last, vec![stepwise[n - 1].clone()]);
    assert!(prefill(&w, &plan, &mut kv_c, &r.tokens[..1], Want::Last).is_err(), "cache is full");
}

fn tiny_q4() -> (Vec<u8>, usize) {
    let bytes = fs::read(format!("{DIR}/tiny-q4.satl")).unwrap();
    let (_, data) = decode_prefix(&bytes).unwrap();
    (bytes, data as usize)
}

#[test]
fn teal_taps_calibrate_mask_and_trace() {
    use atlas_rt::forward::{Taps, prefill_tapped};
    use atlas_rt::teal::{self, DOWN, GATE_UP};

    let r = reference();
    let (bytes, data) = tiny_q4();
    let (header, _) = decode_prefix(&bytes).unwrap();
    let w = weights(&header, &bytes[data..]).unwrap();
    let dense = Plan::dense(&w.config, rope_inv_freq(&w.config).unwrap(), Kernels::Fast);
    let n = r.tokens.len();
    let run = |plan: &Plan, taps: &mut Taps| {
        let mut kv = KvCache::new(&w.config, n);
        prefill_tapped(&w, plan, &mut kv, &r.tokens, Want::All, taps).unwrap()
    };

    // Calibrate on the sequence itself.
    let mut cal = Taps { hist: Some(teal::empty_histograms(w.config.layers)), ..Taps::default() };
    let base = run(&dense, &mut cal);
    let hist = cal.hist.unwrap();

    // Zero thresholds keep every non-zero value: identical logits.
    let zero = Plan { sparsity: Some(teal::thresholds(&hist, 0.0).unwrap()), ..dense.clone() };
    assert_eq!(run(&zero, &mut Taps::default()), base);

    // Target 50%: the first site sees the calibration distribution exactly;
    // later sites drift a little because earlier masks change activations.
    let half = Plan { sparsity: Some(teal::thresholds(&hist, 0.5).unwrap()), ..dense.clone() };
    let mut taps = Taps { trace: Some(Vec::new()), ..Taps::default() };
    let sparse = run(&half, &mut taps);
    assert_ne!(sparse, base);
    for (s, (kept, total)) in taps.kept.iter().enumerate() {
        let eff = 1.0 - *kept as f64 / *total as f64;
        assert!((eff - 0.5).abs() < 0.1, "site {s}: effective sparsity {eff}");
    }

    // Trace: token-major, per layer gate_up then down words; set bits = kept.
    let c = &w.config;
    let words_per_layer = c.hidden / 64 + c.ffn / 64;
    let trace = taps.trace.unwrap();
    assert_eq!(trace.len(), n * c.layers * words_per_layer);
    let bits: u64 = trace.iter().map(|w| w.count_ones() as u64).sum();
    assert_eq!(bits, taps.kept[GATE_UP].0 + taps.kept[DOWN].0);

    // Stepwise decode refuses a sparse plan rather than silently ignoring it.
    let mut kv = KvCache::new(&w.config, n);
    assert!(step(&w, &half, &mut kv, r.tokens[0]).is_err());
}
