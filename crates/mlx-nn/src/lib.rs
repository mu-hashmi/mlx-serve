#![warn(missing_docs)]

//! Neural network layer exports built on top of `mlx-rs`.

/// Re-export of the upstream `mlx-rs::nn` module.
pub mod nn {
    pub use mlx_rs::nn::*;
}

/// Re-export of module traits used by model implementations.
pub mod module {
    pub use mlx_rs::module::{Module, ModuleParameters, ModuleParametersExt};
}
