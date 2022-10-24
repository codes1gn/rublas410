extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_rand;

pub mod blas_executor;
pub mod blas_opcode;
pub mod blas_tensor;

/// Prelude module for users to import
pub mod prelude {
    // helpers
    pub use ndarray::prelude::*;
    pub use ndarray_linalg::*;

    // prelude
    pub use crate::blas_executor::*;
    pub use crate::blas_opcode::*;
    pub use crate::blas_tensor::*;
}

// TODO use custom measurements: TFLOPS for criterion

#[cfg(test)]
mod tests {
    use ndarray::*;
    use ndarray_linalg::*;

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

    #[test]
    fn mat_inv_test() {
        // let a = Array2::<f32>::zeros((64, 32));
        let a: Array2<f32> = random((3, 3));
        a.inv();
    }
}
