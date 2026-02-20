use mlx_rs::{
    Array,
    error::Exception,
    ops::{
        concatenate_axis,
        indexing::{TryIndexMutOp, TryIndexOp},
        zeros_dtype,
    },
};

const KV_CACHE_GROWTH_STEP: i32 = 256;

/// Trait for key-value caches used in autoregressive generation.
pub trait KeyValueCache {
    /// Whether the cache stores quantized KV pairs.
    fn is_quantized(&self) -> bool {
        false
    }

    /// Group size for quantized cache. `None` if not quantized.
    fn group_size(&self) -> Option<i32> {
        None
    }

    /// Bit width for quantized cache. `None` if not quantized.
    fn bits(&self) -> Option<i32> {
        None
    }

    /// Current sequence offset (number of tokens already cached).
    fn offset(&self) -> i32;

    /// Maximum cache size, if bounded.
    fn max_size(&self) -> Option<i32>;

    /// Append new key/value tensors and return the full cached key/value.
    fn update_and_fetch(&mut self, keys: Array, values: Array)
    -> Result<(Array, Array), Exception>;
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn is_quantized(&self) -> bool {
        T::is_quantized(self)
    }

    fn group_size(&self) -> Option<i32> {
        T::group_size(self)
    }

    fn bits(&self) -> Option<i32> {
        T::bits(self)
    }

    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values)
    }
}

/// KV cache that grows in fixed-size blocks and updates slices in place.
#[derive(Debug, Clone, Default)]
pub struct ConcatKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl ConcatKeyValueCache {
    /// Build an empty cache.
    pub fn new() -> Self {
        Self::default()
    }
}

impl KeyValueCache for ConcatKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let keys_shape = keys.shape();
        let values_shape = values.shape();

        if keys_shape.len() < 4 || values_shape.len() < 4 {
            return Err(Exception::custom(
                "key/value tensors must have at least 4 dimensions",
            ));
        }

        let append_len = keys_shape[2];
        if append_len <= 0 {
            return Err(Exception::custom(
                "key/value tensors must have a positive sequence length",
            ));
        }

        let prev = self.offset;
        let required = prev + append_len;
        let capacity = self.keys.as_ref().map_or(0, |arr| arr.shape()[2]);

        if self.keys.is_none() || required > capacity {
            let batch = keys_shape[0];
            let num_heads = keys_shape[1];
            let key_head_dim = keys_shape[3];
            let value_head_dim = values_shape[3];
            let n_steps = (KV_CACHE_GROWTH_STEP + append_len - 1) / KV_CACHE_GROWTH_STEP;
            let grow_tokens = n_steps * KV_CACHE_GROWTH_STEP;
            let key_grow_shape = [batch, num_heads, grow_tokens, key_head_dim];
            let value_grow_shape = [batch, num_heads, grow_tokens, value_head_dim];
            let new_keys = zeros_dtype(&key_grow_shape, keys.dtype())?;
            let new_values = zeros_dtype(&value_grow_shape, values.dtype())?;

            match (self.keys.take(), self.values.take()) {
                (Some(mut existing_keys), Some(mut existing_values)) => {
                    if prev % KV_CACHE_GROWTH_STEP != 0 {
                        existing_keys = existing_keys.try_index((.., .., ..prev, ..))?;
                        existing_values = existing_values.try_index((.., .., ..prev, ..))?;
                    }

                    self.keys = Some(concatenate_axis(&[existing_keys, new_keys], -2)?);
                    self.values = Some(concatenate_axis(&[existing_values, new_values], -2)?);
                }
                _ => {
                    self.keys = Some(new_keys);
                    self.values = Some(new_values);
                }
            }
        }

        self.offset = required;

        {
            let stored_keys = self
                .keys
                .as_mut()
                .ok_or_else(|| Exception::custom("Keys cannot be None after update"))?;
            stored_keys.try_index_mut((.., .., prev..self.offset, ..), &keys)?;
        }
        {
            let stored_values = self
                .values
                .as_mut()
                .ok_or_else(|| Exception::custom("Values cannot be None after update"))?;
            stored_values.try_index_mut((.., .., prev..self.offset, ..), &values)?;
        }

        let result_keys = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("Keys cannot be None after update"))?
            .try_index((.., .., ..self.offset, ..))?;
        let result_values = self
            .values
            .as_ref()
            .ok_or_else(|| Exception::custom("Values cannot be None after update"))?
            .try_index((.., .., ..self.offset, ..))?;

        Ok((result_keys, result_values))
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    /// Create a zero-filled KV pair with shape [1, n_heads, seq_len, head_dim].
    fn make_kv_pair(seq_len: i32, head_dim: i32) -> (Array, Array) {
        let shape = [1, 2, seq_len, head_dim];
        (
            Array::zeros::<f32>(&shape).unwrap(),
            Array::zeros::<f32>(&shape).unwrap(),
        )
    }

    #[test]
    fn test_concat_cache_initial_update() {
        let mut cache = ConcatKeyValueCache::new();
        assert_eq!(cache.offset(), 0);
        assert!(cache.max_size().is_none());
        assert!(!cache.is_quantized());

        let (keys, values) = make_kv_pair(4, 8);
        let (result_keys, result_values) = cache.update_and_fetch(keys, values).unwrap();
        assert_eq!(result_keys.shape(), &[1, 2, 4, 8]);
        assert_eq!(result_values.shape(), &[1, 2, 4, 8]);
        assert_eq!(cache.offset(), 4);
    }

    #[test]
    fn test_concat_cache_sequential_updates() {
        let mut cache = ConcatKeyValueCache::new();

        let (keys1, values1) = make_kv_pair(4, 8);
        cache.update_and_fetch(keys1, values1).unwrap();
        assert_eq!(cache.offset(), 4);

        let (keys2, values2) = make_kv_pair(1, 8);
        let (result_keys, result_values) = cache.update_and_fetch(keys2, values2).unwrap();
        assert_eq!(result_keys.shape(), &[1, 2, 5, 8]);
        assert_eq!(result_values.shape(), &[1, 2, 5, 8]);
        assert_eq!(cache.offset(), 5);
    }

    #[test]
    fn test_concat_cache_many_sequential_updates() {
        let mut cache = ConcatKeyValueCache::new();

        let (keys, values) = make_kv_pair(3, 8);
        cache.update_and_fetch(keys, values).unwrap();
        assert_eq!(cache.offset(), 3);

        for i in 0..5 {
            let (k, v) = make_kv_pair(1, 8);
            let (rk, rv) = cache.update_and_fetch(k, v).unwrap();
            let expected_seq = 3 + i + 1;
            assert_eq!(cache.offset(), expected_seq);
            assert_eq!(rk.shape(), &[1, 2, expected_seq, 8]);
            assert_eq!(rv.shape(), &[1, 2, expected_seq, 8]);
        }

        assert_eq!(cache.offset(), 8);
    }

    #[test]
    fn test_concat_cache_preallocates_and_reuses_capacity() {
        let mut cache = ConcatKeyValueCache::new();

        let (keys, values) = make_kv_pair(3, 8);
        let (result_keys, result_values) = cache.update_and_fetch(keys, values).unwrap();
        assert_eq!(result_keys.shape(), &[1, 2, 3, 8]);
        assert_eq!(result_values.shape(), &[1, 2, 3, 8]);
        assert_eq!(cache.offset(), 3);
        assert_eq!(
            cache.keys.as_ref().unwrap().shape()[2],
            KV_CACHE_GROWTH_STEP
        );
        assert_eq!(
            cache.values.as_ref().unwrap().shape()[2],
            KV_CACHE_GROWTH_STEP
        );

        let (keys2, values2) = make_kv_pair(1, 8);
        cache.update_and_fetch(keys2, values2).unwrap();
        assert_eq!(cache.offset(), 4);
        assert_eq!(
            cache.keys.as_ref().unwrap().shape()[2],
            KV_CACHE_GROWTH_STEP
        );
        assert_eq!(
            cache.values.as_ref().unwrap().shape()[2],
            KV_CACHE_GROWTH_STEP
        );
    }

    #[test]
    fn test_concat_cache_grows_to_next_block() {
        let mut cache = ConcatKeyValueCache::new();

        let (keys, values) = make_kv_pair(KV_CACHE_GROWTH_STEP, 8);
        cache.update_and_fetch(keys, values).unwrap();
        assert_eq!(cache.offset(), KV_CACHE_GROWTH_STEP);
        assert_eq!(
            cache.keys.as_ref().unwrap().shape()[2],
            KV_CACHE_GROWTH_STEP
        );

        let (keys2, values2) = make_kv_pair(1, 8);
        cache.update_and_fetch(keys2, values2).unwrap();
        assert_eq!(cache.offset(), KV_CACHE_GROWTH_STEP + 1);
        assert_eq!(
            cache.keys.as_ref().unwrap().shape()[2],
            KV_CACHE_GROWTH_STEP * 2
        );
        assert_eq!(
            cache.values.as_ref().unwrap().shape()[2],
            KV_CACHE_GROWTH_STEP * 2
        );
    }

    #[test]
    fn test_concat_cache_default_values() {
        let cache = ConcatKeyValueCache::default();
        assert_eq!(cache.offset(), 0);
        assert!(cache.max_size().is_none());
        assert!(!cache.is_quantized());
        assert!(cache.group_size().is_none());
        assert!(cache.bits().is_none());
    }

    #[test]
    fn test_concat_cache_mismatched_shapes_error() {
        let mut cache = ConcatKeyValueCache::new();

        let (keys1, values1) = make_kv_pair(4, 8);
        cache.update_and_fetch(keys1, values1).unwrap();

        // Mismatched head_dim (16 instead of 8)
        let (keys2, values2) = make_kv_pair(1, 16);
        let result = cache.update_and_fetch(keys2, values2);
        assert!(
            result.is_err(),
            "Mismatched head_dim should fail concatenation"
        );
    }

    #[test]
    fn test_concat_cache_1d_keys_error() {
        let mut cache = ConcatKeyValueCache::new();
        let keys = Array::zeros::<f32>(&[4]).unwrap();
        let values = Array::zeros::<f32>(&[4]).unwrap();
        let result = cache.update_and_fetch(keys, values);
        assert!(result.is_err());
    }

    #[test]
    fn test_concat_cache_ref_mut_delegation() {
        let mut cache = ConcatKeyValueCache::new();
        let cache_ref: &mut ConcatKeyValueCache = &mut cache;

        assert_eq!(KeyValueCache::offset(&cache_ref), 0);
        assert!(KeyValueCache::max_size(&cache_ref).is_none());
        assert!(!KeyValueCache::is_quantized(&cache_ref));
        assert!(KeyValueCache::group_size(&cache_ref).is_none());
        assert!(KeyValueCache::bits(&cache_ref).is_none());

        let (keys, values) = make_kv_pair(3, 8);
        let (rk, rv) = cache_ref.update_and_fetch(keys, values).unwrap();
        assert_eq!(rk.shape(), &[1, 2, 3, 8]);
        assert_eq!(rv.shape(), &[1, 2, 3, 8]);
        assert_eq!(KeyValueCache::offset(&cache_ref), 3);
    }
}
