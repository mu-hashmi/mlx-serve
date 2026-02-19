#[path = "support/common.rs"]
mod common;

use std::collections::BTreeSet;

use mlx_engine::model_loader;
use mlx_models::{WeightMapIndex, transformer::ModelArgs};
use mlx_rs::Array;
use safetensors::tensor::SafeTensors;

#[test]
fn check_1_1_config_parsing() {
    let _guard = common::test_lock();
    let model_dir = common::resolve_model_dir();
    let config_raw = std::fs::read_to_string(model_dir.join("config.json"))
        .expect("failed to read config.json");
    let args: ModelArgs = serde_json::from_str(&config_raw).expect("failed to parse config.json");

    assert_eq!(args.hidden_size, 2048, "unexpected hidden_size");
    assert_eq!(
        args.num_attention_heads, 32,
        "unexpected num_attention_heads"
    );
    assert_eq!(
        args.num_hidden_layers, 16,
        "unexpected num_hidden_layers"
    );
    assert_eq!(args.vocab_size, 128_256, "unexpected vocab_size");
}

#[test]
fn check_1_2_weight_loading() {
    let _guard = common::test_lock();
    let model_dir = common::resolve_model_dir();

    let config_raw = std::fs::read_to_string(model_dir.join("config.json"))
        .expect("failed to read config.json");
    let args: ModelArgs = serde_json::from_str(&config_raw).expect("failed to parse config.json");

    let index_path = model_dir.join("model.safetensors.index.json");
    let index_raw = std::fs::read_to_string(&index_path)
        .expect("failed to read model.safetensors.index.json");
    let index: WeightMapIndex =
        serde_json::from_str(&index_raw).expect("failed to parse model.safetensors.index.json");

    let mut shard_paths = BTreeSet::new();
    for shard in index.weight_map.values() {
        shard_paths.insert(model_dir.join(shard));
    }

    let mut loaded_param_count = 0usize;
    for shard_path in &shard_paths {
        let bytes = std::fs::read(shard_path)
            .unwrap_or_else(|_| panic!("failed to read shard {}", shard_path.display()));
        let tensors =
            SafeTensors::deserialize(&bytes).expect("failed to deserialize safetensors shard");
        loaded_param_count = loaded_param_count.saturating_add(tensors.len());
    }

    assert_eq!(
        loaded_param_count,
        index.weight_map.len(),
        "loaded tensor key count does not match index weight_map"
    );

    let norm_key = "model.layers.0.input_layernorm.weight";
    let norm_shard = index
        .weight_map
        .get(norm_key)
        .expect("model.layers.0.input_layernorm.weight not found in weight_map");
    let norm_bytes = std::fs::read(model_dir.join(norm_shard)).expect("failed to read norm shard");
    let norm_tensors =
        SafeTensors::deserialize(&norm_bytes).expect("failed to deserialize norm shard");
    let norm_tensor = norm_tensors
        .tensor(norm_key)
        .expect("failed to load model.layers.0.input_layernorm.weight tensor");

    let expected_shape = [usize::try_from(args.hidden_size).expect("hidden_size out of range")];
    assert_eq!(
        norm_tensor.shape(),
        expected_shape,
        "unexpected input layernorm tensor shape"
    );
}

#[test]
fn check_1_3_full_model_hydration_forward() {
    let _guard = common::test_lock();
    let model_dir = common::resolve_model_dir();

    let config_raw = std::fs::read_to_string(model_dir.join("config.json"))
        .expect("failed to read config.json");
    let args: ModelArgs = serde_json::from_str(&config_raw).expect("failed to parse config.json");

    let mut model = model_loader::load_model(&model_dir).expect("failed to load model");
    let mut cache = model.make_cache();

    let input = Array::from_slice(&[42_i32, 43, 44, 45, 46], &[1, 5]);
    let logits = model
        .forward(&input, None, &mut cache)
        .expect("forward pass failed");

    let expected_shape = [1, 5, args.vocab_size];
    assert_eq!(
        logits.shape(),
        expected_shape,
        "unexpected logits shape after forward pass"
    );
}
