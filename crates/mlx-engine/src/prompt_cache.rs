use std::collections::HashMap;
use std::time::Instant;

use mlx_models::AnyCache;

/// Default page size used for cache accounting.
pub const DEFAULT_CACHE_PAGE_BYTES: usize = 1 << 20;

/// Configuration for the paged prompt cache manager.
#[derive(Debug, Clone, Copy)]
pub struct CacheManagerConfig {
    /// Maximum number of cached prefixes.
    pub max_prefix_entries: usize,
    /// Total cache memory budget in bytes.
    pub max_cache_bytes: usize,
    /// Logical page size used for cache accounting.
    pub page_size_bytes: usize,
    /// Minimum token length required before a prefix is cacheable.
    pub min_prefix_tokens: usize,
}

impl Default for CacheManagerConfig {
    fn default() -> Self {
        Self {
            max_prefix_entries: 8,
            max_cache_bytes: 4 * 1024 * 1024 * 1024,
            page_size_bytes: DEFAULT_CACHE_PAGE_BYTES,
            min_prefix_tokens: 16,
        }
    }
}

/// Error produced by cache reservation/allocation operations.
#[derive(Debug, thiserror::Error)]
pub enum CacheManagerError {
    /// Request exceeds total cache budget.
    #[error(
        "cache request requires {required_bytes} bytes, but cache budget is {max_bytes} bytes"
    )]
    RequestTooLarge {
        /// Estimated bytes needed for this request.
        required_bytes: usize,
        /// Total cache budget.
        max_bytes: usize,
    },

    /// Request cannot be admitted under current pressure.
    #[error(
        "cache pressure too high: need {required_pages} pages and have {free_pages} free pages"
    )]
    InsufficientFreePages {
        /// Number of pages required for admission.
        required_pages: usize,
        /// Number of pages currently free.
        free_pages: usize,
    },

    /// Integer overflow while computing cache footprint.
    #[error("cache accounting overflow")]
    Overflow,
}

/// Opaque request allocation handle for a reserved cache slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(u64);

impl RequestId {
    /// Return the numeric request ID.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Result of reserving cache capacity for a request.
#[derive(Debug, Clone)]
pub struct RequestReservation {
    /// Request identifier used to release capacity after generation.
    pub request_id: RequestId,
    /// Number of prefix tokens reused from cache.
    pub prefix_len: usize,
    /// Cached KV state for the matched prefix.
    pub prefix_cache: Option<AnyCache>,
    /// Token slice that still needs prefill compute.
    pub prefill_tokens: Vec<u32>,
    /// Estimated request cache bytes for `prompt + max_tokens`.
    pub estimated_request_bytes: usize,
}

/// Aggregate memory/capacity counters for observability.
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Bytes currently consumed by active requests.
    pub active_request_bytes: usize,
    /// Bytes currently consumed by cached prefixes.
    pub prefix_bytes: usize,
    /// Total bytes in use (`active + prefix`).
    pub total_used_bytes: usize,
    /// Total bytes available in the page table.
    pub total_capacity_bytes: usize,
    /// Number of free pages.
    pub free_pages: usize,
    /// Number of active requests.
    pub active_requests: usize,
    /// Number of prefix entries.
    pub prefix_entries: usize,
}

/// Paged cache manager for request isolation, prefix sharing, and memory budgeting.
///
/// This manager does metadata/accounting only. Actual KV tensors are stored in
/// `AnyCache` snapshots attached to prefix entries and cloned into requests.
pub struct PromptCacheManager {
    entries: HashMap<u64, PrefixEntry>,
    active_requests: HashMap<RequestId, ActiveRequest>,
    free_pages: Vec<u32>,
    total_pages: usize,
    page_size_bytes: usize,
    max_prefix_entries: usize,
    min_prefix_tokens: usize,
    next_request_id: u64,
}

#[derive(Clone)]
struct PrefixEntry {
    tokens: Vec<u32>,
    cache: AnyCache,
    pages: Vec<u32>,
    estimated_bytes: usize,
    last_accessed: Instant,
    pin_count: usize,
}

struct ActiveRequest {
    owned_pages: Vec<u32>,
    borrowed_prefix: Option<u64>,
    estimated_bytes: usize,
}

impl PromptCacheManager {
    /// Construct a new cache manager using paged accounting.
    pub fn new(config: CacheManagerConfig) -> Self {
        let page_size_bytes = config.page_size_bytes.max(1);
        let total_pages = config.max_cache_bytes.div_ceil(page_size_bytes).max(1);
        let mut free_pages = Vec::with_capacity(total_pages);
        for page in 0..u32::try_from(total_pages).unwrap_or(u32::MAX) {
            free_pages.push(page);
            if free_pages.len() == total_pages {
                break;
            }
        }

        Self {
            entries: HashMap::new(),
            active_requests: HashMap::new(),
            free_pages,
            total_pages,
            page_size_bytes,
            max_prefix_entries: config.max_prefix_entries,
            min_prefix_tokens: config.min_prefix_tokens,
            next_request_id: 1,
        }
    }

    /// Reserve cache capacity for a generation request and return prefix reuse data.
    pub fn begin_request(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        bytes_per_token: usize,
    ) -> Result<RequestReservation, CacheManagerError> {
        let prompt_len = prompt_tokens.len();
        let max_tokens = usize::try_from(max_tokens).map_err(|_| CacheManagerError::Overflow)?;
        let total_tokens = prompt_len
            .checked_add(max_tokens)
            .ok_or(CacheManagerError::Overflow)?;
        let estimated_bytes = total_tokens
            .checked_mul(bytes_per_token)
            .ok_or(CacheManagerError::Overflow)?;

        let total_capacity = self.total_capacity_bytes();
        if estimated_bytes > total_capacity {
            return Err(CacheManagerError::RequestTooLarge {
                required_bytes: estimated_bytes,
                max_bytes: total_capacity,
            });
        }

        let prefix_key = self.find_longest_prefix_key(prompt_tokens);
        let mut prefix_len = 0usize;
        let mut prefix_cache = None;
        let mut borrowed_bytes = 0usize;

        if let Some(key) = prefix_key
            && let Some(entry) = self.entries.get_mut(&key)
        {
            entry.last_accessed = Instant::now();
            entry.pin_count += 1;
            prefix_len = entry.tokens.len();
            borrowed_bytes = entry.estimated_bytes.min(estimated_bytes);
            prefix_cache = Some(entry.cache.clone());
        }

        // If the matched prefix covers the full prompt, we cannot reuse it for
        // next-token logits prefill. Keep the accounting borrow but skip cache reuse.
        let prefill_tokens = if prefix_len > 0 && prefix_len < prompt_tokens.len() {
            prompt_tokens[prefix_len..].to_vec()
        } else {
            prefix_len = 0;
            prefix_cache = None;
            borrowed_bytes = 0;
            prompt_tokens.to_vec()
        };

        let additional_bytes = estimated_bytes.saturating_sub(borrowed_bytes);
        let required_pages = self.pages_for_bytes(additional_bytes);

        self.ensure_free_pages(required_pages)?;
        let owned_pages = self.take_free_pages(required_pages);

        let request_id = RequestId(self.next_request_id);
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or(CacheManagerError::Overflow)?;

        self.active_requests.insert(
            request_id,
            ActiveRequest {
                owned_pages,
                borrowed_prefix: prefix_key,
                estimated_bytes,
            },
        );

        Ok(RequestReservation {
            request_id,
            prefix_len,
            prefix_cache,
            prefill_tokens,
            estimated_request_bytes: estimated_bytes,
        })
    }

    /// Release a previously reserved request slot.
    pub fn finish_request(&mut self, request_id: RequestId) {
        if let Some(active) = self.active_requests.remove(&request_id) {
            self.free_pages.extend(active.owned_pages);
            if let Some(prefix_key) = active.borrowed_prefix
                && let Some(entry) = self.entries.get_mut(&prefix_key)
            {
                entry.pin_count = entry.pin_count.saturating_sub(1);
                entry.last_accessed = Instant::now();
            }
        }
    }

    /// Store a prompt prefix snapshot for future prefix-sharing hits.
    pub fn store_prefix(
        &mut self,
        prompt_tokens: Vec<u32>,
        cache: AnyCache,
        bytes_per_token: usize,
    ) -> Result<(), CacheManagerError> {
        if self.max_prefix_entries == 0 || prompt_tokens.len() < self.min_prefix_tokens {
            return Ok(());
        }

        let estimated_bytes = prompt_tokens
            .len()
            .checked_mul(bytes_per_token)
            .ok_or(CacheManagerError::Overflow)?;
        if estimated_bytes == 0 {
            return Ok(());
        }

        let required_pages = self.pages_for_bytes(estimated_bytes);
        if required_pages > self.total_pages {
            return Ok(());
        }

        let key = hash_tokens(&prompt_tokens);
        if let Some(existing) = self.entries.get_mut(&key)
            && existing.tokens == prompt_tokens
        {
            existing.cache = cache;
            existing.last_accessed = Instant::now();
            return Ok(());
        }

        while self.entries.len() >= self.max_prefix_entries {
            if !self.evict_lru_prefix() {
                return Ok(());
            }
        }

        self.ensure_free_pages(required_pages)?;
        let pages = self.take_free_pages(required_pages);

        self.entries.insert(
            key,
            PrefixEntry {
                tokens: prompt_tokens,
                cache,
                pages,
                estimated_bytes,
                last_accessed: Instant::now(),
                pin_count: 0,
            },
        );

        Ok(())
    }

    /// Remove all unpinned prefix entries.
    pub fn clear_prefixes(&mut self) {
        let removable_keys: Vec<u64> = self
            .entries
            .iter()
            .filter_map(|(key, entry)| (entry.pin_count == 0).then_some(*key))
            .collect();
        for key in removable_keys {
            if let Some(entry) = self.entries.remove(&key) {
                self.free_pages.extend(entry.pages);
            }
        }
    }

    /// Return aggregate cache accounting stats.
    pub fn stats(&self) -> CacheStats {
        let prefix_bytes = self
            .entries
            .values()
            .map(|entry| entry.estimated_bytes)
            .sum::<usize>();
        let active_request_bytes = self
            .active_requests
            .values()
            .map(|active| active.estimated_bytes)
            .sum::<usize>();

        let used_pages = self.total_pages.saturating_sub(self.free_pages.len());
        let total_used_bytes = used_pages.saturating_mul(self.page_size_bytes);

        CacheStats {
            active_request_bytes,
            prefix_bytes,
            total_used_bytes,
            total_capacity_bytes: self.total_capacity_bytes(),
            free_pages: self.free_pages.len(),
            active_requests: self.active_requests.len(),
            prefix_entries: self.entries.len(),
        }
    }

    fn total_capacity_bytes(&self) -> usize {
        self.total_pages.saturating_mul(self.page_size_bytes)
    }

    fn pages_for_bytes(&self, bytes: usize) -> usize {
        if bytes == 0 {
            0
        } else {
            bytes.div_ceil(self.page_size_bytes)
        }
    }

    fn take_free_pages(&mut self, count: usize) -> Vec<u32> {
        let mut pages = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(page_id) = self.free_pages.pop() {
                pages.push(page_id);
            }
        }
        pages
    }

    fn ensure_free_pages(&mut self, required_pages: usize) -> Result<(), CacheManagerError> {
        while self.free_pages.len() < required_pages {
            if !self.evict_lru_prefix() {
                return Err(CacheManagerError::InsufficientFreePages {
                    required_pages,
                    free_pages: self.free_pages.len(),
                });
            }
        }
        Ok(())
    }

    fn evict_lru_prefix(&mut self) -> bool {
        let key = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.pin_count == 0)
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| *key);

        if let Some(key) = key
            && let Some(entry) = self.entries.remove(&key)
        {
            self.free_pages.extend(entry.pages);
            return true;
        }

        false
    }

    fn find_longest_prefix_key(&self, tokens: &[u32]) -> Option<u64> {
        let mut best: Option<(u64, usize)> = None;
        for (key, entry) in &self.entries {
            let len = entry.tokens.len();
            if len < self.min_prefix_tokens || len >= tokens.len() {
                continue;
            }
            if entry
                .tokens
                .iter()
                .zip(tokens.iter())
                .all(|(left, right)| left == right)
            {
                match best {
                    Some((_, best_len)) if len <= best_len => {}
                    _ => best = Some((*key, len)),
                }
            }
        }

        best.map(|(key, _)| key)
    }
}

fn hash_tokens(tokens: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tokens.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use mlx_models::cache::ConcatKeyValueCache;

    fn make_cache(layers: usize) -> AnyCache {
        let kv: Vec<Option<ConcatKeyValueCache>> = (0..layers)
            .map(|_| Some(ConcatKeyValueCache::new()))
            .collect();
        AnyCache::KV(kv)
    }

    fn manager(max_entries: usize, max_mb: usize) -> PromptCacheManager {
        PromptCacheManager::new(CacheManagerConfig {
            max_prefix_entries: max_entries,
            max_cache_bytes: max_mb * 1024 * 1024,
            page_size_bytes: 1024 * 1024,
            min_prefix_tokens: 16,
        })
    }

    #[test]
    fn reservation_rejects_oversized_request() {
        let mut cache = manager(8, 1);
        let prompt: Vec<u32> = (0..256).collect();

        let result = cache.begin_request(&prompt, 4096, 1024);
        assert!(matches!(result, Err(CacheManagerError::RequestTooLarge { .. })));
    }

    #[test]
    fn begin_and_finish_request_releases_pages() {
        let mut cache = manager(8, 4);
        let prompt: Vec<u32> = (0..64).collect();

        let reservation = cache.begin_request(&prompt, 32, 1024).unwrap();
        let during = cache.stats();
        assert_eq!(during.active_requests, 1);
        assert!(during.total_used_bytes > 0);

        cache.finish_request(reservation.request_id);
        let after = cache.stats();
        assert_eq!(after.active_requests, 0);
    }

    #[test]
    fn prefix_hit_reuses_cached_snapshot() {
        let mut cache = manager(8, 8);
        let prefix: Vec<u32> = (0..32).collect();
        cache.store_prefix(prefix.clone(), make_cache(4), 2048).unwrap();

        let mut request = prefix.clone();
        request.extend([100_u32, 101, 102]);
        let reservation = cache.begin_request(&request, 16, 2048).unwrap();

        assert_eq!(reservation.prefix_len, prefix.len());
        assert!(reservation.prefix_cache.is_some());
        assert_eq!(reservation.prefill_tokens, vec![100, 101, 102]);

        cache.finish_request(reservation.request_id);
    }

    #[test]
    fn lru_eviction_happens_under_page_pressure() {
        let mut cache = manager(8, 2);

        let prefix_a: Vec<u32> = (0..64).collect();
        let prefix_b: Vec<u32> = (100..164).collect();
        cache.store_prefix(prefix_a.clone(), make_cache(2), 16_384).unwrap();
        cache.store_prefix(prefix_b.clone(), make_cache(2), 16_384).unwrap();

        let prefix_c: Vec<u32> = (200..264).collect();
        cache.store_prefix(prefix_c.clone(), make_cache(2), 16_384).unwrap();

        let mut request = prefix_c;
        request.extend([999]);
        let hit = cache.begin_request(&request, 8, 16_384).unwrap();
        assert_eq!(hit.prefix_len, 64);
        cache.finish_request(hit.request_id);
    }

    #[test]
    fn clear_prefixes_keeps_active_request_pages_intact() {
        let mut cache = manager(8, 8);

        let prefix: Vec<u32> = (0..64).collect();
        cache.store_prefix(prefix.clone(), make_cache(2), 4096).unwrap();

        let mut request = prefix.clone();
        request.extend([1, 2, 3]);
        let reservation = cache.begin_request(&request, 16, 4096).unwrap();

        cache.clear_prefixes();
        let stats_mid = cache.stats();
        assert_eq!(stats_mid.active_requests, 1);

        cache.finish_request(reservation.request_id);
        cache.clear_prefixes();
        let stats_end = cache.stats();
        assert_eq!(stats_end.prefix_entries, 0);
    }
}
