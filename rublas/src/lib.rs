#[cfg(all(
    feature = "default",
    not(feature = "openblas"),
    not(feature = "netlib")
))]
extern crate ndarray_vanilla as ndarray;

#[cfg(feature = "openblas")]
extern crate ndarray_blas as ndarray;

#[cfg(feature = "netlib")]
extern crate ndarray_blas as ndarray;

pub mod tensor;

/// Prelude module for users to import
///
/// ```
/// use rublas::prelude::*;
///
/// let a = arr2(&[[1, 2, 3], [4, 5, 6]]);
/// let b = arr2(&[[6, 3], [5, 2], [4, 1]]);
/// println!("{}", a.dot(&b));
///
/// let result = 2 + 2;
/// assert_eq!(result, 4);
/// ```
pub mod prelude {
    pub use ndarray::prelude::{arr2, Array1, Array2};
}

// TODO use custom measurements: TFLOPS for criterion

#[cfg(test)]
mod tests {
    use ndarray::prelude::{arr2, Array1, Array2};

    #[test]
    fn it_works() {
        let a = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let b = arr2(&[[6, 3], [5, 2], [4, 1]]);
        println!("{}", a.dot(&b));
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    #[test]
    fn array_builder_test() {
        let a = Array2::<f32>::zeros((64, 32));
        let (m, n) = a.dim();
        let x = Array1::<f32>::zeros(n);
        let y = Array1::<f32>::zeros(m);
        assert_eq!(y.shape(), &[m]);
        assert_eq!(x.shape(), &[n]);
        assert_eq!(a.shape(), &[m, n]);
    }
}
