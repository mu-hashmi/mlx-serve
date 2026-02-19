#[path = "support/common.rs"]
mod common;

use mlx_engine::simple::SamplingConfig;

fn prompt_tokens(engine: &mlx_engine::simple::SimpleEngine, prompt: &str) -> Vec<u32> {
    engine
        .tokenizer()
        .encode(prompt, false)
        .expect("failed to encode prompt")
        .get_ids()
        .to_vec()
}

#[test]
fn check_3_1_greedy_single_token() {
    let _guard = common::test_lock();
    let engine = common::load_engine(4096);

    let prompt = "The capital of France is";
    let tokens = prompt_tokens(&engine, prompt);
    let output = engine
        .generate(&tokens, 1, 0.0, 1.0, &[])
        .expect("generation failed");

    assert!(
        output.text.contains("Par"),
        "expected first token to contain 'Par', got '{}'",
        output.text
    );
}

#[test]
fn check_3_2_greedy_multi_token_deterministic() {
    let _guard = common::test_lock();
    let engine = common::load_engine(4096);

    let prompt = "The capital of France is";
    let tokens = prompt_tokens(&engine, prompt);

    let run1 = engine
        .generate(&tokens, 20, 0.0, 1.0, &[])
        .expect("first generation failed");
    let run2 = engine
        .generate(&tokens, 20, 0.0, 1.0, &[])
        .expect("second generation failed");

    assert_eq!(run1.text, run2.text, "greedy output must be deterministic");
    assert!(
        run1.text.chars().any(|c| c.is_ascii_alphabetic()),
        "expected recognizable English text, got '{}'",
        run1.text
    );
}

#[test]
fn check_3_3_sampled_generation_seed_control() {
    let _guard = common::test_lock();
    let engine = common::load_engine(4096);

    let prompt = "Once upon a";
    let tokens = prompt_tokens(&engine, prompt);

    let cfg_seed_42 = SamplingConfig {
        temperature: 1.5,
        top_p: 1.0,
        top_k: Some(200),
        repetition_penalty: 1.0,
        seed: Some(42),
    };

    let run1 = engine
        .generate_with_sampling(&tokens, 50, cfg_seed_42, &[])
        .expect("seed=42 generation run1 failed");
    let run2 = engine
        .generate_with_sampling(&tokens, 50, cfg_seed_42, &[])
        .expect("seed=42 generation run2 failed");

    assert_eq!(run1.text, run2.text, "same seed should be deterministic");

    let cfg_seed_43 = SamplingConfig {
        seed: Some(43),
        ..cfg_seed_42
    };
    let run3 = engine
        .generate_with_sampling(&tokens, 50, cfg_seed_43, &[])
        .expect("seed=43 generation failed");

    assert_ne!(
        run1.text, run3.text,
        "different seeds should produce different sampled output"
    );
}

#[test]
fn check_3_4_stop_conditions() {
    let _guard = common::test_lock();
    let engine = common::load_engine(4096);

    let prompt = "List three colors:";
    let tokens = prompt_tokens(&engine, prompt);

    let capped = engine
        .generate(&tokens, 10, 0.0, 1.0, &[])
        .expect("max_tokens generation failed");
    assert_eq!(
        capped.completion_tokens, 10,
        "max_tokens=10 should emit exactly 10 completion tokens"
    );

    let eos_prompt = "Question: What is 2+2?\nAnswer using only one token and then stop.";
    let eos_tokens = prompt_tokens(&engine, eos_prompt);
    let eos_run = engine
        .generate(&eos_tokens, 500, 0.0, 1.0, &[])
        .expect("eos generation failed");

    assert!(
        eos_run.completion_tokens < 500,
        "expected EOS before max_tokens, got completion_tokens={} with finish_reason='{}'",
        eos_run.completion_tokens,
        eos_run.finish_reason
    );
}
