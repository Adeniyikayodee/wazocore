//! Converts a tiny random Llama checkpoint with the real binary, then checks
//! `inspect` and `verify` against it.

use std::collections::HashMap;
use std::fs;
use std::process::Command;

use atlas_convert::format::{self, Kind, ModelConfig};
use atlas_convert::plan::{self, Options};
use half::bf16;
use safetensors::tensor::{Dtype, TensorView};

fn tiny() -> ModelConfig {
    ModelConfig {
        hidden: 128,
        ffn: 256,
        layers: 2,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 64,
        vocab: 96,
        rms_norm_eps: 1e-5,
        rope_theta: 500_000.0,
        rope_scaling: None,
        rope_layout: "hf_rotate_half".into(),
    }
}

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_atlas-convert"))
}

#[test]
fn converts_inspects_and_verifies_a_tiny_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = tiny();
    let config = serde_json::json!({
        "hidden_size": cfg.hidden, "intermediate_size": cfg.ffn, "num_hidden_layers": cfg.layers,
        "num_attention_heads": cfg.n_heads, "num_key_value_heads": cfg.n_kv_heads,
        "vocab_size": cfg.vocab, "rms_norm_eps": cfg.rms_norm_eps, "rope_theta": cfg.rope_theta
    });
    fs::write(dir.path().join("config.json"), config.to_string()).unwrap();

    let shapes = plan::required_shapes(&cfg, &Options::default());
    let data: HashMap<String, Vec<u8>> = shapes
        .iter()
        .enumerate()
        .map(|(k, (name, shape))| {
            let n: usize = shape.iter().product();
            let bytes = (0..n)
                .flat_map(|i| bf16::from_f32(((i * 7 + k * 13) as f32 * 0.011).sin() * 0.08).to_le_bytes())
                .collect();
            (name.clone(), bytes)
        })
        .collect();
    let views: Vec<(String, TensorView)> = shapes
        .iter()
        .map(|(name, shape)| (name.clone(), TensorView::new(Dtype::BF16, shape.clone(), &data[name]).unwrap()))
        .collect();
    safetensors::serialize_to_file(views, None, &dir.path().join("model.safetensors")).unwrap();

    let out = dir.path().join("tiny.satl");
    let status = bin()
        .args(["convert", "--model-dir"])
        .arg(dir.path())
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());

    let bytes = fs::read(&out).unwrap();
    let (header, data_start) = format::decode_prefix(&bytes).unwrap();
    assert_eq!(data_start % format::PAGE, 0);
    assert_eq!(header.config.head_dim, 64);
    assert_eq!(header.source.files[0].sha256.len(), 64);
    let last = header.tensors.last().unwrap();
    assert_eq!(bytes.len() as u64, data_start + last.offset + last.length);
    let gate_up = header.tensors.iter().find(|t| t.name == "layers.1.mlp.gate_up").unwrap();
    assert!(matches!(gate_up.kind, Kind::Columns { stride: 4096, .. }));

    let inspect = bin().arg("inspect").arg(&out).output().unwrap();
    assert!(String::from_utf8_lossy(&inspect.stdout).contains("layers.0.mlp.gate_up"));

    let verify = bin()
        .args(["verify", "--every", "1", "--model-dir"])
        .arg(dir.path())
        .arg("--satl")
        .arg(&out)
        .output()
        .unwrap();
    assert!(verify.status.success());
    let report = String::from_utf8_lossy(&verify.stdout);
    let worst: f64 = report.lines().last().unwrap().split_whitespace().last().unwrap().parse().unwrap();
    assert!(worst < 0.12, "worst rel_rmse {worst}\n{report}");
}
