use ndarray::{Array, ArrayBase, Axis, Data, Dimension, RemoveAxis, Zip};
use num_traits::Float;
use ordered_float::OrderedFloat;

/// Calculate a maximum along an axis.
pub fn maxf<T, S, D>(a: &ArrayBase<S, D>, axis: usize) -> Array<T, <D as Dimension>::Smaller>
where
    T: Float,
    S: Data<Elem = T>,
    D: Dimension + RemoveAxis,
{
    Zip::from(a.lanes(Axis(axis)))
        .map_collect(|v| *v.iter().max_by_key(|&&s| OrderedFloat(s)).unwrap())
}

/// Calculate a maximum along an axis and keep the same dimensions.
pub fn maxf_keep_dims<D, T, S, Smaller>(a: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
where
    T: Float,
    S: Data<Elem = T>,
    D: Dimension<Smaller = Smaller> + RemoveAxis,
    Smaller: Dimension<Larger = D>,
{
    let max = maxf(a, axis);
    max.insert_axis(Axis(axis))
}

/// Maxf computes the max on a float array along an axis.
pub trait Maxf<T, D>
where
    D: Dimension,
{
    /// Compute the max along an axis.
    fn maxf(&self, axis: usize) -> Array<T, <D as Dimension>::Smaller>;
}

impl<T, S, D> Maxf<T, D> for ArrayBase<S, D>
where
    T: Float,
    S: Data<Elem = T>,
    D: Dimension + RemoveAxis,
{
    fn maxf(&self, axis: usize) -> Array<T, <D as Dimension>::Smaller> {
        maxf(self, axis)
    }
}

/// Compute the max along an axis and keep the same dimensions.
pub trait MaxfKeepDims<T, D>
where
    D: Dimension,
{
    /// Compute the max along an axis and keep the same dimensions.
    fn maxf_keep_dims(&self, axis: usize) -> Array<T, D>;
}

impl<T, S, D, Smaller> MaxfKeepDims<T, D> for ArrayBase<S, D>
where
    T: Float,
    S: Data<Elem = T>,
    D: Dimension<Smaller = Smaller> + RemoveAxis,
    Smaller: Dimension<Larger = D>,
{
    fn maxf_keep_dims(&self, axis: usize) -> Array<T, D> {
        maxf_keep_dims(self, axis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxf() {
        let a = Array::from_shape_vec((2, 3), vec![2.2, 4.4, 1.5, 2.4, 0.0, 0.0]).unwrap();

        let max = maxf(&a, 1);

        assert_eq!(max.shape(), &[2]);

        assert_eq!(max[0], 4.4);
        assert_eq!(max[1], 2.4);
    }

    #[test]
    fn test_maxf_keep_dims() {
        let a = Array::from_shape_vec((2, 3), vec![2.2, 4.4, 1.5, 2.4, 0.0, 0.0]).unwrap();

        let max = maxf_keep_dims(&a, 1);

        assert_eq!(max.shape(), &[2, 1]);

        assert_eq!(max, Array::from_shape_vec((2, 1), vec![4.4, 2.4]).unwrap());
    }
}
