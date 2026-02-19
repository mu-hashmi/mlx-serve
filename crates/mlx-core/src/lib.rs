#![warn(missing_docs)]

//! Core MLX abstractions and utilities.

use mlx_rs::{Array, error::Exception};
use mlx_sys::{MetalDeviceInfo, MlxSysError};

/// Re-export of the upstream `mlx-rs` crate.
pub mod mlx {
    pub use mlx_rs::*;
}

/// Runtime memory helpers backed by MLX C APIs.
pub mod memory {
    pub use mlx_sys::{
        MetalDeviceInfo, MlxSysError, clear_cache, get_active_memory_bytes, get_cache_memory_bytes,
        metal_device_info, set_cache_limit, set_wired_limit,
    };
}

/// Evaluate one or more arrays.
pub fn eval<'a>(arrays: impl IntoIterator<Item = &'a Array>) -> Result<(), Exception> {
    mlx_rs::transforms::eval(arrays)
}

/// Create a tensor of ones.
pub fn ones<T: mlx_rs::ArrayElement>(shape: &[i32]) -> Result<Array, Exception> {
    Array::ones::<T>(shape)
}

/// Create a tensor of zeros.
pub fn zeros<T: mlx_rs::ArrayElement>(shape: &[i32]) -> Result<Array, Exception> {
    Array::zeros::<T>(shape)
}

/// Read active MLX memory usage in bytes.
pub fn active_memory_bytes() -> Result<usize, MlxSysError> {
    mlx_sys::get_active_memory_bytes()
}

/// Read cached MLX memory usage in bytes.
pub fn cache_memory_bytes() -> Result<usize, MlxSysError> {
    mlx_sys::get_cache_memory_bytes()
}

/// Clear MLX cache memory.
pub fn clear_mlx_cache() -> Result<(), MlxSysError> {
    mlx_sys::clear_cache()
}

/// Read metal device metadata from MLX.
pub fn metal_info() -> MetalDeviceInfo {
    mlx_sys::metal_device_info()
}
