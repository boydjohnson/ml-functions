//! Ops are mathematical operations on ndarray Arrays.

mod exp;
mod max;
mod sigmoid;
mod softmax;

pub use exp::{exp, Exp};
pub use max::{maxf, maxf_keep_dims, Maxf, MaxfKeepDims};
pub use sigmoid::{sigmoid, Sigmoid};
pub use softmax::softmax;
