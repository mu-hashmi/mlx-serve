#[path = "support/common.rs"]
mod common;

use mlx_engine::error::EngineError;

fn tokens(engine: &mlx_engine::simple::SimpleEngine, prompt: &str) -> Vec<u32> {
    engine
        .tokenizer()
        .encode(prompt, false)
        .expect("failed to encode prompt")
        .get_ids()
        .to_vec()
}

#[test]
fn check_4_1_cache_isolation_between_requests() {
    let _guard = common::test_lock();

    let engine = common::load_engine(4096);

    let prompt1 = "Translate to French: Hello";
    let prompt2 = "What is 2+2? Answer with one number.";

    let prompt1_tokens = tokens(&engine, prompt1);
    let prompt2_tokens = tokens(&engine, prompt2);

    let _first = engine
        .generate(&prompt1_tokens, 20, 0.0, 1.0, &[])
        .expect("first request failed");
    let second_after_first = engine
        .generate(&prompt2_tokens, 20, 0.0, 1.0, &[])
        .expect("second request failed");

    let lower = second_after_first.text.to_lowercase();
    for french_marker in ["bonjour", "salut", "fran", "merci"] {
        assert!(
            !lower.contains(french_marker),
            "second response leaked prior prompt context: '{}'",
            second_after_first.text
        );
    }

    let fresh_engine = common::load_engine(4096);
    let prompt2_tokens_fresh = tokens(&fresh_engine, prompt2);
    let second_fresh = fresh_engine
        .generate(&prompt2_tokens_fresh, 20, 0.0, 1.0, &[])
        .expect("fresh second request failed");

    assert_eq!(
        second_after_first.text, second_fresh.text,
        "second request output differs between sequential and fresh process"
    );
}

#[test]
fn check_4_2_memory_reclamation() {
    let _guard = common::test_lock();

    let engine = common::load_engine(4096);
    let long_prompt = common::long_prompt_tokens(&engine, 2200);

    let _before = engine
        .active_memory_bytes()
        .expect("failed to read active memory before generation");

    let output = engine
        .generate(&long_prompt, 64, 0.0, 1.0, &[])
        .expect("long prompt generation failed");
    std::mem::drop(output);

    let after_generation = engine
        .active_memory_bytes()
        .expect("failed to read active memory after generation");
    let cache_bytes = engine
        .prefix_cache_bytes()
        .expect("failed to read prefix cache bytes");

    assert!(cache_bytes > 0, "expected non-zero cached prefix bytes");

    engine
        .clear_runtime_cache()
        .expect("failed to clear runtime cache");

    let after_clear = engine
        .active_memory_bytes()
        .expect("failed to read active memory after clear");

    let reclaimed = after_generation.saturating_sub(after_clear);
    assert!(
        reclaimed >= cache_bytes / 2,
        "expected at least 50% cache reclamation, reclaimed={} cache_bytes={} after_gen={} after_clear={}",
        reclaimed,
        cache_bytes,
        after_generation,
        after_clear
    );
}

#[test]
fn check_4_3_memory_limit_enforcement() {
    let _guard = common::test_lock();

    let constrained_engine = common::load_engine(50);
    let prompt = common::long_prompt_tokens(&constrained_engine, 2200);

    let result = constrained_engine.generate(&prompt, 256, 0.0, 1.0, &[]);
    match result {
        Err(EngineError::Generation(msg)) => {
            assert!(
                msg.contains("cache request requires") || msg.contains("cache pressure"),
                "unexpected memory limit error message: {msg}"
            );
        }
        Err(other) => panic!("unexpected error type: {other}"),
        Ok(output) => panic!(
            "expected memory limit failure, but generation succeeded with {} tokens",
            output.completion_tokens
        ),
    }
}
