#![warn(missing_docs)]
#![allow(unsafe_code)]

//! Raw MLX C bindings and low-level memory/runtime helpers.

use thiserror::Error;

/// Re-export of the upstream generated `mlx-sys` bindings.
#[doc(inline)]
pub use mlx_sys_upstream::*;

const SUCCESS: i32 = 0;

/// Errors from low-level MLX runtime calls.
#[derive(Debug, Error)]
pub enum MlxSysError {
    /// The underlying MLX C API returned a non-zero status code.
    #[error("MLX call failed with status code {0}")]
    Status(i32),
}

/// Metal device information reported by MLX.
#[derive(Debug, Clone, Copy)]
pub struct MetalDeviceInfo {
    /// Maximum recommended Metal working set in bytes.
    pub max_recommended_working_set_size: usize,
    /// Reported device memory size in bytes.
    pub memory_size: usize,
}

/// Read active MLX memory usage in bytes.
pub fn get_active_memory_bytes() -> Result<usize, MlxSysError> {
    let mut value = 0usize;
    let status = unsafe { mlx_get_active_memory(&mut value as *mut usize) };
    if status == SUCCESS {
        Ok(value)
    } else {
        Err(MlxSysError::Status(status))
    }
}

/// Read cached MLX memory usage in bytes.
pub fn get_cache_memory_bytes() -> Result<usize, MlxSysError> {
    let mut value = 0usize;
    let status = unsafe { mlx_get_cache_memory(&mut value as *mut usize) };
    if status == SUCCESS {
        Ok(value)
    } else {
        Err(MlxSysError::Status(status))
    }
}

/// Clear MLX cached memory buffers.
pub fn clear_cache() -> Result<(), MlxSysError> {
    let status = unsafe { mlx_clear_cache() };
    if status == SUCCESS {
        Ok(())
    } else {
        Err(MlxSysError::Status(status))
    }
}

/// Set the MLX cache memory limit in bytes.
pub fn set_cache_limit(limit: usize) -> Result<usize, MlxSysError> {
    let mut previous = 0usize;
    let status = unsafe { mlx_set_cache_limit(&mut previous as *mut usize, limit) };
    if status == SUCCESS {
        Ok(previous)
    } else {
        Err(MlxSysError::Status(status))
    }
}

/// Set wired memory limit for Metal residency in bytes.
pub fn set_wired_limit(limit: usize) -> Result<usize, MlxSysError> {
    let mut previous = 0usize;
    let status = unsafe { mlx_set_wired_limit(&mut previous as *mut usize, limit) };
    if status == SUCCESS {
        Ok(previous)
    } else {
        Err(MlxSysError::Status(status))
    }
}

/// Read metal device information from MLX.
pub fn metal_device_info() -> MetalDeviceInfo {
    let info = unsafe { mlx_metal_device_info() };
    MetalDeviceInfo {
        max_recommended_working_set_size: info.max_recommended_working_set_size,
        memory_size: info.memory_size,
    }
}
