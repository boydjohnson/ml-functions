use ndarray::{Array, ArrayBase, Data, Dimension};
use num_traits::Float;

pub fn exp<T, S, D>(x: ArrayBase<S, D>) -> Array<T, D>
where
    T: Float,
    S: Data<Elem = T>,
    D: Dimension,
{
    x.map(|v| v.exp())
}
