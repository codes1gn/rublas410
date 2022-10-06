use crate::blas_tensor::{BlasTensor, TensorKind};
use crate::prelude::Array2;
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
        match lhs.data {
            TensorKind::FloatMatrix(ref _lhs) => match rhs.data {
                TensorKind::FloatMatrix(ref _rhs) => {
                    let mut out_data = Array2::<f32>::zeros([lhs.shape[0], rhs.shape[1]]);
                    general_mat_mul(1.0, &_lhs, &_rhs, 1.0, &mut out_data);
                    let mut out = BlasTensor {
                        data: TensorKind::from(out_data),
                        shape: vec![lhs.shape[0], rhs.shape[1]],
                    };
                    return out;
                }
                _ => panic!("rhs operand's type not compatible with return type"),
            },
            _ => panic!("lhs operand's type not supported"),
        }
    }

    // gemm with normal-layout mat * normal-layout mat; also consumes operands ownerships
    pub fn gemm_nn(&self, lhs: &BlasTensor, rhs: &BlasTensor, out: &mut BlasTensor) -> () {
        match out.data {
            TensorKind::FloatMatrix(ref mut _out) => match lhs.data {
                TensorKind::FloatMatrix(ref _lhs) => match rhs.data {
                    TensorKind::FloatMatrix(ref _rhs) => {
                        general_mat_mul(1.0, _lhs, _rhs, 1.0, _out);
                    }
                    _ => panic!("rhs operand's type not compatible with return type"),
                },
                _ => panic!("lhs operand's type not compatible with return type"),
            },
            _ => panic!("return type not supported for this gemm"),
        }
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_gemm_nn_owned_2d() {
        let a = BlasTensor::ones(vec![17, 23]);
        let b = BlasTensor::ones(vec![23, 18]);
        let exec = BlasExecutor::new();
        let c = exec.gemm_nn_owned(a, b);
        let cref = BlasTensor::from_vec_shape([23.0; 17 * 18].to_vec(), vec![17, 18]);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [17, 18]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_gemm_nn_2d() {
        let a = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let b = BlasTensor::ones(vec![3, 2]);
        let mut c = BlasTensor::zeros(vec![2, 2]);
        let cref = BlasTensor::from_vec_shape(vec![6.6000004, 6.6000004, 16.5, 16.5], vec![2, 2]);

        let exec = BlasExecutor::new();
        exec.gemm_nn(&a, &b, &mut c);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 2]);
        assert_eq!(c, cref);
    }
}
