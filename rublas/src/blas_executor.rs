use crate::blas_tensor::{BlasTensor, TensorKind};
use ndarray::linalg::general_mat_mul;
use ndarray::linalg::general_mat_vec_mul;

pub struct BlasExecutor {}

impl BlasExecutor {
    pub fn new() -> Self {
        Self {}
    }

    // TODO add type check
    // gemm with normal-layout mat * normal-layout mat; also consumes operands ownerships
    pub fn gemm_nn_owned(&self, lhs: BlasTensor, rhs: BlasTensor) -> BlasTensor {
        let out = BlasTensor::zeros(vec![lhs.shape[0], rhs.shape[1]]);
        out
    }

    // gemm with normal-layout mat * normal-layout mat; also consumes operands ownerships
    pub fn gemm_nn(&self, lhs: &BlasTensor, rhs: &BlasTensor, out: &mut BlasTensor) -> () {
        *out = BlasTensor::zeros(vec![lhs.shape[0], rhs.shape[1]]);
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_gemm_nn_owned_2d() {
        let a = BlasTensor::zeros(vec![17, 23]);
        let b = BlasTensor::zeros(vec![23, 18]);
        let exec = BlasExecutor::new();
        let c = exec.gemm_nn_owned(a, b);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [17, 18]);
    }

    #[test]
    fn test_gemm_nn_2d() {
        let a = BlasTensor::zeros(vec![17, 23]);
        let b = BlasTensor::zeros(vec![23, 18]);
        let mut c = BlasTensor::zeros(vec![17, 18]);

        let exec = BlasExecutor::new();
        exec.gemm_nn(&a, &b, &mut c);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [17, 18]);
    }
}
