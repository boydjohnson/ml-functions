//! Ops are mathematical operations on ndarray Arrays.

pub mod argmax;
pub mod argsort;
pub mod exp;
pub mod max;
pub mod sigmoid;
pub mod softmax;
pub mod sort;

pub use argmax::{argmaxf, argmaxf_keep_dims, Argmaxf, ArgmaxfKeepDims};
pub use argsort::{argsortf, Argsortf};
pub use exp::{exp, Exp};
pub use max::{maxf, maxf_keep_dims, Maxf, MaxfKeepDims};
pub use sigmoid::{sigmoid, Sigmoid};
pub use softmax::{softmax, Softmax};
