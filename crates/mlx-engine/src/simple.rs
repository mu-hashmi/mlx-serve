use std::path::Path;
use std::sync::Mutex;

use mlx_models::{AnyCache, AnyModel, qwen3_next::Qwen3NextModelArgs};
use mlx_rs::{
    Array,
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tokenizers::Tokenizer;

use crate::{
    chat_template::{ChatMessage, ChatTemplateRenderer},
    engine::{GenerationOutput, StreamingOutput},
    error::EngineError,
    model_loader,
    prompt_cache::{
        CacheManagerConfig, CacheManagerError, DEFAULT_CACHE_PAGE_BYTES, PromptCacheManager,
        RequestId, RequestReservation,
    },
};

/// Default maximum number of cached prefixes.
const DEFAULT_PREFIX_CACHE_SIZE: usize = 8;
/// Default maximum cache budget in bytes (4 GiB).
const DEFAULT_MAX_CACHE_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Runtime options for the simple inference engine.
#[derive(Debug, Clone)]
pub struct SimpleEngineOptions {
    /// Maximum total KV cache memory budget for active requests and shared prefixes.
    pub max_cache_bytes: usize,
    /// Maximum number of cached prompt prefixes.
    pub prefix_cache_entries: usize,
    /// Logical cache page size in bytes.
    pub cache_page_bytes: usize,
    /// Minimum prefix length eligible for prefix sharing.
    pub min_prefix_tokens: usize,
}

impl Default for SimpleEngineOptions {
    fn default() -> Self {
        Self {
            max_cache_bytes: DEFAULT_MAX_CACHE_BYTES,
            prefix_cache_entries: DEFAULT_PREFIX_CACHE_SIZE,
            cache_page_bytes: DEFAULT_CACHE_PAGE_BYTES,
            min_prefix_tokens: 16,
        }
    }
}

/// Sampling controls for token selection.
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    /// Softmax temperature.
    pub temperature: f32,
    /// Nucleus sampling threshold.
    pub top_p: f32,
    /// Optional top-k cutoff.
    pub top_k: Option<usize>,
    /// Repetition penalty (`1.0` disables it).
    pub repetition_penalty: f32,
    /// Optional deterministic RNG seed.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: 1.0,
            seed: None,
        }
    }
}

enum SamplerRng {
    Seeded(Box<ChaCha8Rng>),
    Thread(rand::rngs::ThreadRng),
}

impl SamplerRng {
    fn from_seed(seed: Option<u64>) -> Self {
        match seed {
            Some(value) => Self::Seeded(Box::new(ChaCha8Rng::seed_from_u64(value))),
            None => Self::Thread(rand::rng()),
        }
    }

    fn random_f64(&mut self) -> f64 {
        match self {
            SamplerRng::Seeded(rng) => rng.random::<f64>(),
            SamplerRng::Thread(rng) => rng.random::<f64>(),
        }
    }
}

/// Inference engine with request-isolated KV reservations and shared prefix cache pages.
pub struct SimpleEngine {
    model_template: Mutex<AnyModel>,
    cache_manager: Mutex<PromptCacheManager>,
    tokenizer: Tokenizer,
    template: ChatTemplateRenderer,
    model_name: String,
    eos_token_ids: Vec<u32>,
    max_cache_bytes: usize,
    bytes_per_token: usize,
}

/// Intermediate request state after cache reservation and model clone.
struct PreparedGeneration {
    request_id: RequestId,
    model: AnyModel,
    cache: AnyCache,
    prompt_array: Array,
    prompt_len: u32,
}

struct RequestLease<'a> {
    engine: &'a SimpleEngine,
    request_id: RequestId,
}

impl<'a> RequestLease<'a> {
    fn new(engine: &'a SimpleEngine, request_id: RequestId) -> Self {
        Self { engine, request_id }
    }
}

impl Drop for RequestLease<'_> {
    fn drop(&mut self) {
        self.engine.release_request(self.request_id);
    }
}

/// Why generation terminated.
enum FinishCondition {
    Eos,
    StopSequence(String),
    MaxTokens,
    None,
}

impl FinishCondition {
    fn is_finished(&self) -> bool {
        !matches!(self, Self::None)
    }

    fn reason_str(&self) -> Option<&'static str> {
        match self {
            Self::Eos | Self::StopSequence(_) => Some("stop"),
            Self::MaxTokens => Some("length"),
            Self::None => None,
        }
    }
}

impl SimpleEngine {
    /// Load a model and tokenizer from a local path or HF repo ID.
    pub fn load(source: impl AsRef<Path>) -> Result<Self, EngineError> {
        Self::load_with_options(source, SimpleEngineOptions::default())
    }

    /// Load a model and tokenizer with explicit engine options.
    pub fn load_with_options(
        source: impl AsRef<Path>,
        options: SimpleEngineOptions,
    ) -> Result<Self, EngineError> {
        let model_dir = model_loader::resolve_model_dir(&source)?;
        let model_name = source
            .as_ref()
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_owned());

        tracing::info!(model_dir = %model_dir.display(), "Loading model");

        let model = model_loader::load_model(&model_dir)?;
        let tokenizer = model_loader::load_tokenizer(&model_dir)?;
        let template = ChatTemplateRenderer::from_model_dir(&model_dir)?;
        let bytes_per_token = estimate_bytes_per_token(&model)?;

        let eos_token_ids = extract_eos_tokens(&model_dir);

        tracing::info!(
            model_name = %model_name,
            eos_tokens = ?eos_token_ids,
            bytes_per_token,
            max_cache_bytes = options.max_cache_bytes,
            page_bytes = options.cache_page_bytes,
            prefix_entries = options.prefix_cache_entries,
            "Engine ready"
        );

        let cache_manager = PromptCacheManager::new(CacheManagerConfig {
            max_prefix_entries: options.prefix_cache_entries,
            max_cache_bytes: options.max_cache_bytes,
            page_size_bytes: options.cache_page_bytes,
            min_prefix_tokens: options.min_prefix_tokens,
        });

        Ok(Self {
            model_template: Mutex::new(model),
            cache_manager: Mutex::new(cache_manager),
            tokenizer,
            template,
            model_name,
            eos_token_ids,
            max_cache_bytes: options.max_cache_bytes,
            bytes_per_token,
        })
    }

    /// Return the configured maximum cache budget in bytes.
    pub fn max_cache_bytes(&self) -> usize {
        self.max_cache_bytes
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get a reference to the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Read active MLX memory usage in bytes.
    pub fn active_memory_bytes(&self) -> Result<usize, EngineError> {
        mlx_core::active_memory_bytes()
            .map_err(|error| EngineError::Generation(format!("failed to read MLX memory: {error}")))
    }

    /// Read current shared-prefix cache usage in bytes.
    pub fn prefix_cache_bytes(&self) -> Result<usize, EngineError> {
        let manager = self
            .cache_manager
            .lock()
            .map_err(|error| EngineError::Generation(format!("Cache lock poisoned: {error}")))?;
        Ok(manager.stats().prefix_bytes)
    }

    /// Clear prefix cache entries and release MLX cache buffers.
    pub fn clear_runtime_cache(&self) -> Result<(), EngineError> {
        {
            let mut manager = self
                .cache_manager
                .lock()
                .map_err(|error| EngineError::Generation(format!("Cache lock poisoned: {error}")))?;
            manager.clear_prefixes();
        }
        mlx_core::clear_mlx_cache()
            .map_err(|error| EngineError::Generation(format!("failed to clear MLX cache: {error}")))
    }

    /// Apply chat template and tokenize messages.
    pub fn prepare_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<Vec<u32>, EngineError> {
        let prompt = self.template.apply(messages, tools, true)?;
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| EngineError::Tokenization(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Convert prompt length to u32, returning a descriptive error on overflow.
    fn prompt_len(prompt_tokens: &[u32]) -> Result<u32, EngineError> {
        prompt_tokens
            .len()
            .try_into()
            .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))
    }

    fn map_cache_manager_error(error: CacheManagerError) -> EngineError {
        EngineError::Generation(error.to_string())
    }

    fn clone_model_template(&self) -> Result<AnyModel, EngineError> {
        let model = self
            .model_template
            .lock()
            .map_err(|error| EngineError::Generation(format!("Model lock poisoned: {error}")))?;
        Ok(model.clone())
    }

    fn release_request(&self, request_id: RequestId) {
        match self.cache_manager.lock() {
            Ok(mut manager) => manager.finish_request(request_id),
            Err(error) => {
                tracing::error!(
                    request_id = request_id.as_u64(),
                    error = %error,
                    "Cache manager lock poisoned while releasing request"
                );
            }
        }
    }

    /// Reserve request cache capacity and resolve tokens to feed into prefill.
    fn prepare_generation(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
    ) -> Result<PreparedGeneration, EngineError> {
        let prompt_len = Self::prompt_len(prompt_tokens)?;

        let reservation = {
            let mut manager = self
                .cache_manager
                .lock()
                .map_err(|e| EngineError::Generation(format!("Cache lock poisoned: {e}")))?;
            manager
                .begin_request(prompt_tokens, max_tokens, self.bytes_per_token)
                .map_err(Self::map_cache_manager_error)?
        };

        let model = self.clone_model_template()?;

        let RequestReservation {
            request_id,
            prefix_len,
            prefix_cache,
            prefill_tokens,
            estimated_request_bytes,
        } = reservation;

        if prefix_len > 0 {
            tracing::debug!(
                prefix_len,
                total_len = prompt_tokens.len(),
                estimated_request_bytes,
                "Reusing cached prefix"
            );
        }

        let cache = prefix_cache.unwrap_or_else(|| model.make_cache());
        let prompt_array = Array::from(prefill_tokens.as_slice()).index(NewAxis);

        if prompt_array.shape().contains(&0) {
            return Err(EngineError::Generation(
                "internal error: prefill token buffer is empty".to_owned(),
            ));
        }

        Ok(PreparedGeneration {
            request_id,
            model,
            cache,
            prompt_array,
            prompt_len,
        })
    }

    /// Run the prefill forward pass and sample the first token. Stores the
    /// post-prefill KV state back into the prefix cache.
    fn run_prefill(
        &self,
        prompt_tokens: &[u32],
        prepared: &mut PreparedGeneration,
        sampling: SamplingConfig,
        rng: &mut SamplerRng,
    ) -> Result<Array, EngineError> {
        let logits = prepared
            .model
            .forward(&prepared.prompt_array, None, &mut prepared.cache)
            .map_err(EngineError::Mlx)?;
        let sampled = sample_token_id(&logits.index((.., -1, ..)), sampling, prompt_tokens, rng)?;
        let current_token = Array::from_slice(&[sampled], &[1]);
        eval([&current_token]).map_err(EngineError::Mlx)?;

        let mut manager = self
            .cache_manager
            .lock()
            .map_err(|e| EngineError::Generation(format!("Cache lock poisoned: {e}")))?;
        if let Err(error) =
            manager.store_prefix(prompt_tokens.to_vec(), prepared.cache.clone(), self.bytes_per_token)
        {
            tracing::warn!(error = %error, "Prefix cache insert skipped");
        }

        Ok(current_token)
    }

    /// Decode a single step: forward pass on the current token and sample the next.
    fn decode_step(
        current_token: &Array,
        model: &mut AnyModel,
        cache: &mut AnyCache,
        sampling: SamplingConfig,
        history_tokens: &[u32],
        rng: &mut SamplerRng,
    ) -> Result<Array, EngineError> {
        let decode_input = current_token.index((.., NewAxis));
        let logits = model
            .forward(&decode_input, None, cache)
            .map_err(EngineError::Mlx)?;
        let sampled = sample_token_id(&logits.index((.., -1, ..)), sampling, history_tokens, rng)?;
        Ok(Array::from_slice(&[sampled], &[1]))
    }

    /// Decode tokens, check EOS/stop/max, and return the appropriate finish condition.
    fn check_termination(
        &self,
        token_id: u32,
        completion_len: u32,
        max_tokens: u32,
        decoded_text: &str,
        stop_sequences: &[String],
    ) -> FinishCondition {
        if self.eos_token_ids.contains(&token_id) {
            return FinishCondition::Eos;
        }
        if !stop_sequences.is_empty()
            && let Some(truncated) = check_stop_sequences(decoded_text, stop_sequences)
        {
            return FinishCondition::StopSequence(truncated);
        }
        if completion_len >= max_tokens {
            return FinishCondition::MaxTokens;
        }
        FinishCondition::None
    }

    /// Decode the token buffer and return the text, mapping tokenizer errors.
    fn decode_tokens(&self, tokens: &[u32]) -> Result<String, EngineError> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| EngineError::Tokenization(e.to_string()))
    }

    /// Convert a token count to u32, with an overflow error.
    fn completion_len(tokens: &[u32]) -> Result<u32, EngineError> {
        tokens
            .len()
            .try_into()
            .map_err(|_| EngineError::Generation("Too many tokens generated".to_owned()))
    }

    /// Generate a complete response from a token prompt.
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        stop_sequences: &[String],
    ) -> Result<GenerationOutput, EngineError> {
        self.generate_with_sampling(
            prompt_tokens,
            max_tokens,
            SamplingConfig {
                temperature,
                top_p,
                ..SamplingConfig::default()
            },
            stop_sequences,
        )
    }

    /// Generate a complete response with full sampling controls.
    pub fn generate_with_sampling(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        sampling: SamplingConfig,
        stop_sequences: &[String],
    ) -> Result<GenerationOutput, EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation("Prompt is empty".to_owned()));
        }
        if max_tokens == 0 {
            return Ok(GenerationOutput {
                text: String::new(),
                finish_reason: "length".to_owned(),
                prompt_tokens: Self::prompt_len(prompt_tokens)?,
                completion_tokens: 0,
            });
        }

        let mut rng = SamplerRng::from_seed(sampling.seed);
        let mut prepared = self.prepare_generation(prompt_tokens, max_tokens)?;
        let _lease = RequestLease::new(self, prepared.request_id);
        let prompt_len = prepared.prompt_len;
        let mut current_token = self.run_prefill(prompt_tokens, &mut prepared, sampling, &mut rng)?;

        let mut history_tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut tokens: Vec<u32> = Vec::new();
        let first_token_id: u32 = current_token.item();
        tokens.push(first_token_id);
        history_tokens.push(first_token_id);

        let first_decoded = self.decode_tokens(&tokens)?;

        let condition = self.check_termination(
            first_token_id,
            1,
            max_tokens,
            &first_decoded,
            stop_sequences,
        );

        if condition.is_finished() {
            let text = match &condition {
                FinishCondition::StopSequence(truncated) => truncated.clone(),
                _ => first_decoded,
            };
            return Ok(GenerationOutput {
                text,
                finish_reason: condition.reason_str().unwrap_or("stop").to_owned(),
                prompt_tokens: prompt_len,
                completion_tokens: 1,
            });
        }

        // Decode loop
        loop {
            current_token = Self::decode_step(
                &current_token,
                &mut prepared.model,
                &mut prepared.cache,
                sampling,
                &history_tokens,
                &mut rng,
            )?;

            let token_id: u32 = current_token.item();
            tokens.push(token_id);
            history_tokens.push(token_id);

            if tokens.len().is_multiple_of(32) {
                eval([&current_token]).map_err(EngineError::Mlx)?;
            }

            let completion_len = Self::completion_len(&tokens)?;
            let text = self.decode_tokens(&tokens)?;

            let loop_condition =
                self.check_termination(token_id, completion_len, max_tokens, &text, stop_sequences);

            if loop_condition.is_finished() {
                let final_text = match &loop_condition {
                    FinishCondition::StopSequence(truncated) => truncated.clone(),
                    _ => text,
                };
                return Ok(GenerationOutput {
                    text: final_text,
                    finish_reason: loop_condition.reason_str().unwrap_or("stop").to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                });
            }
        }
    }

    /// Generate tokens one at a time, sending each via the provided channel.
    ///
    /// If the receiver is dropped (client disconnected), generation stops early.
    pub fn generate_streaming(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        stop_sequences: &[String],
        sender: tokio::sync::mpsc::Sender<StreamingOutput>,
    ) -> Result<(), EngineError> {
        self.generate_streaming_with_sampling(
            prompt_tokens,
            max_tokens,
            SamplingConfig {
                temperature,
                top_p,
                ..SamplingConfig::default()
            },
            stop_sequences,
            sender,
        )
    }

    /// Streaming generation with full sampling controls.
    pub fn generate_streaming_with_sampling(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        sampling: SamplingConfig,
        stop_sequences: &[String],
        sender: tokio::sync::mpsc::Sender<StreamingOutput>,
    ) -> Result<(), EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation("Prompt is empty".to_owned()));
        }
        if max_tokens == 0 {
            let prompt_len = Self::prompt_len(prompt_tokens)?;
            let _ = sender.blocking_send(StreamingOutput {
                new_text: String::new(),
                finished: true,
                finish_reason: Some("length".to_owned()),
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
            return Ok(());
        }

        let mut rng = SamplerRng::from_seed(sampling.seed);
        let mut prepared = self.prepare_generation(prompt_tokens, max_tokens)?;
        let _lease = RequestLease::new(self, prepared.request_id);
        let prompt_len = prepared.prompt_len;
        let mut current_token = self.run_prefill(prompt_tokens, &mut prepared, sampling, &mut rng)?;

        let mut history_tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut all_tokens: Vec<u32> = Vec::new();
        let first_token_id: u32 = current_token.item();
        all_tokens.push(first_token_id);
        history_tokens.push(first_token_id);

        let first_decoded = self.decode_tokens(&all_tokens)?;
        let (first_text, first_hit_stop) = if !stop_sequences.is_empty() {
            if let Some(truncated) = check_stop_sequences(&first_decoded, stop_sequences) {
                (truncated, true)
            } else {
                (first_decoded.clone(), false)
            }
        } else {
            (first_decoded.clone(), false)
        };
        let mut prev_decoded_len = first_decoded.len();

        let first_is_eos = self.eos_token_ids.contains(&first_token_id);
        let finished = first_is_eos || first_hit_stop || 1 >= max_tokens;

        if sender
            .blocking_send(StreamingOutput {
                new_text: first_text,
                finished,
                finish_reason: if first_is_eos || first_hit_stop {
                    Some("stop".to_owned())
                } else if 1 >= max_tokens {
                    Some("length".to_owned())
                } else {
                    None
                },
                prompt_tokens: prompt_len,
                completion_tokens: 1,
            })
            .is_err()
        {
            return Ok(());
        }

        if finished {
            return Ok(());
        }

        // Decode loop
        loop {
            current_token = Self::decode_step(
                &current_token,
                &mut prepared.model,
                &mut prepared.cache,
                sampling,
                &history_tokens,
                &mut rng,
            )?;

            let token_id: u32 = current_token.item();
            all_tokens.push(token_id);
            history_tokens.push(token_id);

            if all_tokens.len().is_multiple_of(32) {
                eval([&current_token]).map_err(EngineError::Mlx)?;
            }

            let completion_len = Self::completion_len(&all_tokens)?;

            let full_text = self.decode_tokens(&all_tokens)?;
            let new_text = full_text
                .get(prev_decoded_len..)
                .unwrap_or_default()
                .to_owned();
            let old_decoded_len = prev_decoded_len;
            prev_decoded_len = full_text.len();

            let (final_new_text, hit_stop_seq) = if !stop_sequences.is_empty() {
                if let Some(truncated) = check_stop_sequences(&full_text, stop_sequences) {
                    let emit = truncated
                        .get(old_decoded_len..)
                        .unwrap_or_default()
                        .to_owned();
                    (emit, true)
                } else {
                    (new_text, false)
                }
            } else {
                (new_text, false)
            };

            let is_eos = self.eos_token_ids.contains(&token_id);
            let is_max = completion_len >= max_tokens;
            let step_finished = is_eos || is_max || hit_stop_seq;

            let finish_reason = if is_eos || hit_stop_seq {
                Some("stop".to_owned())
            } else if is_max {
                Some("length".to_owned())
            } else {
                None
            };

            if sender
                .blocking_send(StreamingOutput {
                    new_text: final_new_text,
                    finished: step_finished,
                    finish_reason,
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                })
                .is_err()
            {
                return Ok(());
            }

            if step_finished {
                break;
            }
        }

        Ok(())
    }
}

fn positive_i32_to_usize(name: &str, value: i32) -> Result<usize, EngineError> {
    usize::try_from(value).map_err(|_| {
        EngineError::Generation(format!("invalid {name} in model config: {value}"))
    })
}

fn sample_token_id(
    logits: &Array,
    sampling: SamplingConfig,
    history_tokens: &[u32],
    rng: &mut SamplerRng,
) -> Result<u32, EngineError> {
    if logits.ndim() == 0 {
        return Err(EngineError::Generation(
            "logits tensor must have at least one dimension".to_owned(),
        ));
    }

    let float_logits = logits.as_type::<f32>().map_err(EngineError::Mlx)?;
    let mut scores = float_logits.as_slice::<f32>().to_vec();
    if scores.is_empty() {
        return Err(EngineError::Generation(
            "logits tensor is unexpectedly empty".to_owned(),
        ));
    }

    if (sampling.repetition_penalty - 1.0).abs() > f32::EPSILON {
        for token in history_tokens {
            if let Ok(index) = usize::try_from(*token)
                && let Some(value) = scores.get_mut(index)
            {
                if *value < 0.0 {
                    *value *= sampling.repetition_penalty;
                } else {
                    *value /= sampling.repetition_penalty;
                }
            }
        }
    }

    if sampling.temperature <= 0.0 {
        let mut best_index = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, value) in scores.iter().enumerate() {
            if *value > best_value {
                best_value = *value;
                best_index = index;
            }
        }
        return u32::try_from(best_index).map_err(|_| {
            EngineError::Generation("token index overflow during greedy sampling".to_owned())
        });
    }

    if let Some(top_k) = sampling.top_k
        && top_k > 0 && top_k < scores.len()
    {
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let threshold = sorted[top_k - 1];
        for score in &mut scores {
            if *score < threshold {
                *score = f32::NEG_INFINITY;
            }
        }
    }

    let max_score = scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f64> = scores
        .iter()
        .map(|score| ((*score - max_score) / sampling.temperature).exp() as f64)
        .collect();

    if sampling.top_p < 1.0 {
        let top_p = f64::from(sampling.top_p.clamp(0.0, 1.0));
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_by(|a, b| probs[*b].total_cmp(&probs[*a]));

        if top_p <= 0.0 {
            let best = order[0];
            for (index, prob) in probs.iter_mut().enumerate() {
                if index != best {
                    *prob = 0.0;
                }
            }
        } else {
            let mut cumulative = 0.0f64;
            let mut kept = 0usize;
            for index in order {
                if kept > 0 && cumulative >= top_p {
                    probs[index] = 0.0;
                    continue;
                }
                cumulative += probs[index];
                kept = kept.saturating_add(1);
            }
        }
    }

    let total: f64 = probs.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(EngineError::Generation(
            "sampling produced invalid probability distribution".to_owned(),
        ));
    }

    let draw = rng.random_f64();
    let mut cumulative = 0.0f64;
    for (index, prob) in probs.iter().enumerate() {
        cumulative += prob / total;
        if draw <= cumulative {
            return u32::try_from(index).map_err(|_| {
                EngineError::Generation("token index overflow during sampling".to_owned())
            });
        }
    }

    let last = probs.len().saturating_sub(1);
    u32::try_from(last).map_err(|_| {
        EngineError::Generation("token index overflow during sampling".to_owned())
    })
}

fn bytes_per_token_from_transformer_args(args: &mlx_models::transformer::ModelArgs) -> Result<usize, EngineError> {
    let layers = positive_i32_to_usize("num_hidden_layers", args.num_hidden_layers)?;
    let kv_heads = positive_i32_to_usize("num_key_value_heads", args.num_key_value_heads)?;
    let hidden = positive_i32_to_usize("hidden_size", args.hidden_size)?;
    let heads = positive_i32_to_usize("num_attention_heads", args.num_attention_heads)?;
    if heads == 0 || hidden % heads != 0 {
        return Err(EngineError::Generation(format!(
            "invalid attention dimensions: hidden_size={} num_attention_heads={}",
            args.hidden_size, args.num_attention_heads
        )));
    }
    let head_dim = hidden / heads;
    let bytes_per_elem = 2usize;
    let per_layer = kv_heads
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_mul(bytes_per_elem))
        .ok_or_else(|| EngineError::Generation("KV cache size overflow".to_owned()))?;
    per_layer
        .checked_mul(layers)
        .ok_or_else(|| EngineError::Generation("KV cache size overflow".to_owned()))
}

fn bytes_per_token_from_qwen3_next_args(args: &Qwen3NextModelArgs) -> Result<usize, EngineError> {
    let layers = positive_i32_to_usize("num_hidden_layers", args.num_hidden_layers)?;
    let kv_heads = positive_i32_to_usize("num_key_value_heads", args.num_key_value_heads)?;
    let head_dim = positive_i32_to_usize("head_dim", args.head_dim)?;
    let bytes_per_elem = 2usize;
    let per_layer = kv_heads
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_mul(bytes_per_elem))
        .ok_or_else(|| EngineError::Generation("KV cache size overflow".to_owned()))?;
    per_layer
        .checked_mul(layers)
        .ok_or_else(|| EngineError::Generation("KV cache size overflow".to_owned()))
}

fn estimate_bytes_per_token(model: &AnyModel) -> Result<usize, EngineError> {
    match model {
        AnyModel::Transformer(transformer) => bytes_per_token_from_transformer_args(&transformer.args),
        AnyModel::Qwen3Next(qwen3_next) => bytes_per_token_from_qwen3_next_args(&qwen3_next.args),
    }
}

/// Check if any stop sequence appears in the generated text.
/// Returns Some(truncated_text) if a stop sequence was found, None otherwise.
fn check_stop_sequences(text: &str, stop_sequences: &[String]) -> Option<String> {
    let mut earliest: Option<usize> = None;
    for seq in stop_sequences {
        if let Some(pos) = text.find(seq.as_str()) {
            earliest = Some(earliest.map_or(pos, |prev| prev.min(pos)));
        }
    }
    earliest.map(|pos| text.get(..pos).unwrap_or_default().to_owned())
}

/// Extract EOS token IDs from config.json.
fn extract_eos_tokens(model_dir: &Path) -> Vec<u32> {
    let config_path = model_dir.join("config.json");
    let config_str = match std::fs::read_to_string(&config_path) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(path = %config_path.display(), error = %e, "Could not read config.json for EOS tokens");
            return vec![];
        }
    };

    let config: serde_json::Value = match serde_json::from_str(&config_str) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "Could not parse config.json for EOS tokens");
            return vec![];
        }
    };

    match config.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .map_or_else(Vec::new, |id| vec![id]),
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_u64().and_then(|val| u32::try_from(val).ok()))
            .collect(),
        Some(other) => {
            tracing::warn!(value = ?other, "Unexpected eos_token_id type in config.json");
            vec![]
        }
        None => {
            tracing::warn!(
                "No eos_token_id found in config.json, generation will rely on max_tokens"
            );
            vec![]
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::check_stop_sequences;

    /// Write a config.json file into the given directory with the provided JSON content.
    fn write_config(dir: &std::path::Path, json: &str) {
        std::fs::write(dir.join("config.json"), json).unwrap();
    }

    /// Create a temp dir, write config.json with the given content, and return
    /// the result of `extract_eos_tokens`.
    fn eos_from_config(json: &str) -> Vec<u32> {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), json);
        super::extract_eos_tokens(dir.path())
    }

    #[test]
    fn test_single_stop_sequence_found() {
        let result = check_stop_sequences("Hello world, goodbye!", &["goodbye".to_owned()]);
        assert_eq!(result, Some("Hello world, ".to_owned()));
    }

    #[test]
    fn test_no_stop_sequence_match() {
        let stops = vec!["goodbye".to_owned(), "farewell".to_owned()];
        assert!(check_stop_sequences("Hello world", &stops).is_none());
    }

    #[test]
    fn test_empty_stop_sequences_list() {
        assert!(check_stop_sequences("Hello world", &[]).is_none());
    }

    #[test]
    fn test_empty_text() {
        assert!(check_stop_sequences("", &["hello".to_owned()]).is_none());
    }

    #[test]
    fn test_stop_sequence_at_beginning() {
        let result = check_stop_sequences("STOP rest of text", &["STOP".to_owned()]);
        assert_eq!(result, Some(String::new()));
    }

    #[test]
    fn test_stop_sequence_at_end() {
        let result = check_stop_sequences("Hello world END", &["END".to_owned()]);
        assert_eq!(result, Some("Hello world ".to_owned()));
    }

    fn assert_stop_sequence(text: &str, stops: &[&str], expected: &str) {
        let owned_stops: Vec<String> = stops.iter().map(|s| (*s).to_owned()).collect();
        let result = check_stop_sequences(text, &owned_stops);
        assert_eq!(result, Some(expected.to_owned()));
    }

    #[test]
    fn test_multiple_stop_sequences_earliest_wins() {
        assert_stop_sequence("aaa bbb ccc ddd", &["ccc", "bbb"], "aaa ");
    }

    #[test]
    fn test_multiple_stop_sequences_earliest_wins_reverse_order() {
        assert_stop_sequence("aaa bbb ccc ddd", &["bbb", "ccc"], "aaa ");
    }

    #[test]
    fn test_overlapping_stop_sequences_prefix() {
        // "ab" is a prefix of "abc". "ab" appears first at position 0.
        let stops = vec!["abc".to_owned(), "ab".to_owned()];
        assert_eq!(check_stop_sequences("abc def", &stops), Some(String::new()));
    }

    #[test]
    fn test_stop_sequence_appears_multiple_times() {
        let result = check_stop_sequences("before stop middle stop after", &["stop".to_owned()]);
        assert_eq!(result, Some("before ".to_owned()));
    }

    #[test]
    fn test_stop_sequence_is_entire_text() {
        assert_eq!(
            check_stop_sequences("STOP", &["STOP".to_owned()]),
            Some(String::new())
        );
    }

    #[test]
    fn test_stop_sequence_with_newlines() {
        let result = check_stop_sequences("line one\nline two\nline three", &["\n".to_owned()]);
        assert_eq!(result, Some("line one".to_owned()));
    }

    #[test]
    fn test_extract_eos_tokens_single_number() {
        assert_eq!(eos_from_config(r#"{"eos_token_id": 151643}"#), vec![151643]);
    }

    #[test]
    fn test_extract_eos_tokens_array() {
        assert_eq!(
            eos_from_config(r#"{"eos_token_id": [151643, 151645]}"#),
            vec![151643, 151645]
        );
    }

    #[test]
    fn test_extract_eos_tokens_missing_field() {
        assert!(eos_from_config(r#"{"model_type": "qwen2"}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_unexpected_type() {
        assert!(eos_from_config(r#"{"eos_token_id": "string"}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_missing_config_file() {
        let dir = tempfile::tempdir().unwrap();
        assert!(super::extract_eos_tokens(dir.path()).is_empty());
    }

    // -- Additional check_stop_sequences edge cases --

    #[test]
    fn test_stop_sequence_substring_of_another() {
        assert_stop_sequence("Hello stop_now world", &["stop_now", "stop"], "Hello ");
    }

    #[test]
    fn test_stop_sequence_unicode() {
        let stops = vec!["\u{1F600}".to_owned()];
        assert!(check_stop_sequences("Hello world, a]b stop here", &stops).is_none());

        let result = check_stop_sequences("Hello \u{1F600} world", &stops);
        assert_eq!(result, Some("Hello ".to_owned()));
    }

    #[test]
    fn test_stop_sequence_unicode_multibyte() {
        let stops = vec!["arr\u{00EA}t".to_owned()];
        let result = check_stop_sequences("Bonjour le monde, arr\u{00EA}t ici", &stops);
        assert_eq!(result, Some("Bonjour le monde, ".to_owned()));
    }

    #[test]
    fn test_stop_sequence_very_long_text_short_stop() {
        let long_text = "a".repeat(10_000) + "STOP" + &"b".repeat(5_000);
        let result = check_stop_sequences(&long_text, &["STOP".to_owned()]);
        assert_eq!(result, Some("a".repeat(10_000)));
    }

    // -- Additional extract_eos_tokens edge cases --

    #[test]
    fn test_extract_eos_tokens_float_value() {
        // serde_json parses 151643.0 as a float, and as_u64() returns None for floats
        assert!(eos_from_config(r#"{"eos_token_id": 151643.0}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_string_value() {
        assert!(eos_from_config(r#"{"eos_token_id": "not_a_number"}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_nested_array() {
        // Inner arrays are not numbers, so as_u64() returns None for them
        assert!(eos_from_config(r#"{"eos_token_id": [[1, 2], [3, 4]]}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_negative_number() {
        // as_u64() returns None for negative numbers
        assert!(eos_from_config(r#"{"eos_token_id": -1}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_very_large_number() {
        // u32::MAX is 4294967295; as_u64() succeeds but u32::try_from fails
        assert!(eos_from_config(r#"{"eos_token_id": 4294967296}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_empty_array() {
        assert!(eos_from_config(r#"{"eos_token_id": []}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_mixed_types_in_array() {
        // Only numeric entries are extracted; "two" is skipped
        assert_eq!(
            eos_from_config(r#"{"eos_token_id": [1, "two", 3]}"#),
            vec![1, 3]
        );
    }
}
