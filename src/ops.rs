//! Ops are mathematical operations on ndarray Arrays.

pub mod exp;
pub mod max;
pub mod sigmoid;
pub mod softmax;

pub use exp::{exp, Exp};
pub use max::{maxf, maxf_keep_dims, Maxf, MaxfKeepDims};
pub use sigmoid::{sigmoid, Sigmoid};
pub use softmax::softmax;
