//! Integration checks for MLX FFI sanity.

use mlx_core::mlx::Array;
use std::sync::{Mutex, MutexGuard, OnceLock};

fn mlx_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("test lock poisoned")
}

#[test]
fn ffi_ones_roundtrip() {
    let _guard = mlx_test_lock();
    let tensor = Array::ones::<f32>(&[2, 3]).expect("failed to allocate ones tensor");
    tensor.eval().expect("failed to evaluate tensor");

    let values = tensor.as_slice::<f32>();
    assert_eq!(values.len(), 6, "expected exactly six values");
    assert!(
        values.iter().all(|value| (*value - 1.0).abs() < 1e-6),
        "expected every element to be 1.0, got {values:?}"
    );
}

#[test]
fn ffi_matmul_shape_and_values() {
    let _guard = mlx_test_lock();
    let a = Array::from_slice(
        &[
            0.5_f32, -1.0, 2.0, 0.0, 1.5, -0.5, 0.25, 3.0, 1.0, 0.75, -2.0, 4.0, -1.0, 2.0,
            0.5, -0.25, 2.5, 1.0, 0.0, -3.0, 1.0, 0.5, -1.5, 2.0, -0.5, 0.0, 1.0, 0.5, -2.5,
            3.5, -1.0, 1.5,
        ],
        &[4, 8],
    );
    let b = Array::from_slice(
        &[
            1.0_f32, 0.0, -1.0, 2.0, 0.5, -0.5, 1.5, 0.0, -1.5, 2.0, 0.0, 1.0, 0.25, -0.75,
            2.5, -1.0, 1.0, 0.5, -0.5, 3.0, -2.0, 1.0, 0.75, -0.25, 1.25, -1.5, 2.0, 0.5, 0.0,
            1.0, -2.0, 0.75,
        ],
        &[8, 4],
    );

    let product = a.matmul(&b).expect("matmul failed");
    product.eval().expect("matmul evaluation failed");

    assert_eq!(product.shape(), &[4, 4]);

    let values = product.as_slice::<f32>();
    assert!(
        values.iter().any(|value| value.abs() > 1e-6),
        "matmul output unexpectedly all zeros: {values:?}"
    );
}
