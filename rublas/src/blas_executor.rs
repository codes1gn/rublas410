use ndarray::linalg::general_mat_mul;
use ndarray::linalg::general_mat_vec_mul;

use crate::blas_opcode::BlasOpCode;
use crate::blas_tensor::{BlasTensor, TensorKind};
use crate::prelude::Array2;

#[derive(Debug)]
pub struct BlasExecutor {}

impl BlasExecutor {
    pub fn new() -> Self {
        Self {}
    }

    pub fn binary_compute_owned(
        &self,
        op: BlasOpCode,
        lhs: BlasTensor,
        rhs: BlasTensor,
    ) -> BlasTensor {
        match op {
            BlasOpCode::AddF => self.addf32_owned(lhs, rhs),
            BlasOpCode::SubF => self.subf32_owned(lhs, rhs),
            BlasOpCode::MulF => self.mulf32_owned(lhs, rhs),
            BlasOpCode::DivF => self.divf32_owned(lhs, rhs),
            BlasOpCode::GemmF => self.gemm_owned(lhs, rhs),
            _ => panic!("not wired opcode"),
        }
    }

    pub fn binary_compute_side_effect(
        &self,
        op: BlasOpCode,
        lhs: &BlasTensor,
        rhs: &BlasTensor,
        out: &mut BlasTensor,
    ) {
        match op {
            BlasOpCode::AddF => self.addf32_side_effect(lhs, rhs, out),
            BlasOpCode::SubF => self.subf32_side_effect(lhs, rhs, out),
            BlasOpCode::MulF => self.mulf32_side_effect(lhs, rhs, out),
            BlasOpCode::DivF => self.divf32_side_effect(lhs, rhs, out),
            BlasOpCode::GemmF => self.gemm_side_effect(lhs, rhs, out),
            _ => panic!("not wired opcode"),
        }
    }

    pub fn addf32_owned(&self, lhs: BlasTensor, rhs: BlasTensor) -> BlasTensor {
        match lhs.data {
            TensorKind::FloatVector(ref _lhs) => match rhs.data {
                TensorKind::FloatVector(ref _rhs) => {
                    let out_data = _lhs + _rhs;
                    let mut out = BlasTensor {
                        data: TensorKind::from(out_data),
                        shape: vec![lhs.shape[0]],
                    };
                    return out;
                }
                _ => panic!("rhs operand's type not compatible with return type"),
            },
            TensorKind::FloatMatrix(ref _lhs) => match rhs.data {
                TensorKind::FloatMatrix(ref _rhs) => {
                    let out_data = _lhs + _rhs;
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

    pub fn subf32_owned(&self, lhs: BlasTensor, rhs: BlasTensor) -> BlasTensor {
        match lhs.data {
            TensorKind::FloatVector(ref _lhs) => match rhs.data {
                TensorKind::FloatVector(ref _rhs) => {
                    let out_data = _lhs - _rhs;
                    let mut out = BlasTensor {
                        data: TensorKind::from(out_data),
                        shape: vec![lhs.shape[0]],
                    };
                    return out;
                }
                _ => panic!("rhs operand's type not compatible with return type"),
            },
            TensorKind::FloatMatrix(ref _lhs) => match rhs.data {
                TensorKind::FloatMatrix(ref _rhs) => {
                    let out_data = _lhs - _rhs;
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

    pub fn mulf32_owned(&self, lhs: BlasTensor, rhs: BlasTensor) -> BlasTensor {
        match lhs.data {
            TensorKind::FloatVector(ref _lhs) => match rhs.data {
                TensorKind::FloatVector(ref _rhs) => {
                    let out_data = _lhs * _rhs;
                    let mut out = BlasTensor {
                        data: TensorKind::from(out_data),
                        shape: vec![lhs.shape[0]],
                    };
                    return out;
                }
                _ => panic!("rhs operand's type not compatible with return type"),
            },
            TensorKind::FloatMatrix(ref _lhs) => match rhs.data {
                TensorKind::FloatMatrix(ref _rhs) => {
                    let out_data = _lhs * _rhs;
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

    pub fn divf32_owned(&self, lhs: BlasTensor, rhs: BlasTensor) -> BlasTensor {
        match lhs.data {
            TensorKind::FloatVector(ref _lhs) => match rhs.data {
                TensorKind::FloatVector(ref _rhs) => {
                    let out_data = _lhs / _rhs;
                    let mut out = BlasTensor {
                        data: TensorKind::from(out_data),
                        shape: vec![lhs.shape[0]],
                    };
                    return out;
                }
                _ => panic!("rhs operand's type not compatible with return type"),
            },
            TensorKind::FloatMatrix(ref _lhs) => match rhs.data {
                TensorKind::FloatMatrix(ref _rhs) => {
                    let out_data = _lhs / _rhs;
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

    pub fn addf32_side_effect(&self, lhs: &BlasTensor, rhs: &BlasTensor, out: &mut BlasTensor) {
        match lhs.data {
            TensorKind::FloatVector(ref _lhs) => match rhs.data {
                TensorKind::FloatVector(ref _rhs) => {
                    let out_data = _lhs + _rhs;
                    *out = BlasTensor {
                        data: TensorKind::from(out_data),
                        shape: vec![lhs.shape[0]],
                    };
                }
                _ => panic!("rhs operand's type not compatible with return type"),
            },
            TensorKind::FloatMatrix(ref _lhs) => match rhs.data {
                TensorKind::FloatMatrix(ref _rhs) => {
                    let out_data = _lhs + _rhs;
                    *out = BlasTensor {
                        data: TensorKind::from(out_data),
                        shape: vec![lhs.shape[0], rhs.shape[1]],
                    };
                }
                _ => panic!("rhs operand's type not compatible with return type"),
            },
            _ => panic!("lhs operand's type not supported"),
        }
    }

    pub fn subf32_side_effect(&self, lhs: &BlasTensor, rhs: &BlasTensor, out: &mut BlasTensor) {
        match lhs.data {
            TensorKind::FloatVector(ref _lhs) => match rhs.data {
                TensorKind::FloatVector(ref _rhs) => {
                    let out_data = _lhs - _rhs;
                    *out = BlasTensor {
                        data: TensorKind::from(out_data),
                        shape: vec![lhs.shape[0]],
                    };
                }
                _ => panic!("rhs operand's type not compatible with return type"),
            },
            TensorKind::FloatMatrix(ref _lhs) => match rhs.data {
                TensorKind::FloatMatrix(ref _rhs) => {
                    let out_data = _lhs - _rhs;
                    *out = BlasTensor {
                        data: TensorKind::from(out_data),
                        shape: vec![lhs.shape[0], rhs.shape[1]],
                    };
                }
                _ => panic!("rhs operand's type not compatible with return type"),
            },
            _ => panic!("lhs operand's type not supported"),
        }
    }

    pub fn mulf32_side_effect(&self, lhs: &BlasTensor, rhs: &BlasTensor, out: &mut BlasTensor) {
        match lhs.data {
            TensorKind::FloatVector(ref _lhs) => match rhs.data {
                TensorKind::FloatVector(ref _rhs) => {
                    let out_data = _lhs * _rhs;
                    *out = BlasTensor {
                        data: TensorKind::from(out_data),
                        shape: vec![lhs.shape[0]],
                    };
                }
                _ => panic!("rhs operand's type not compatible with return type"),
            },
            TensorKind::FloatMatrix(ref _lhs) => match rhs.data {
                TensorKind::FloatMatrix(ref _rhs) => {
                    let out_data = _lhs * _rhs;
                    *out = BlasTensor {
                        data: TensorKind::from(out_data),
                        shape: vec![lhs.shape[0], rhs.shape[1]],
                    };
                }
                _ => panic!("rhs operand's type not compatible with return type"),
            },
            _ => panic!("lhs operand's type not supported"),
        }
    }

    pub fn divf32_side_effect(&self, lhs: &BlasTensor, rhs: &BlasTensor, out: &mut BlasTensor) {
        match lhs.data {
            TensorKind::FloatVector(ref _lhs) => match rhs.data {
                TensorKind::FloatVector(ref _rhs) => {
                    let out_data = _lhs / _rhs;
                    *out = BlasTensor {
                        data: TensorKind::from(out_data),
                        shape: vec![lhs.shape[0]],
                    };
                }
                _ => panic!("rhs operand's type not compatible with return type"),
            },
            TensorKind::FloatMatrix(ref _lhs) => match rhs.data {
                TensorKind::FloatMatrix(ref _rhs) => {
                    let out_data = _lhs / _rhs;
                    *out = BlasTensor {
                        data: TensorKind::from(out_data),
                        shape: vec![lhs.shape[0], rhs.shape[1]],
                    };
                }
                _ => panic!("rhs operand's type not compatible with return type"),
            },
            _ => panic!("lhs operand's type not supported"),
        }
    }

    // TODO add type check
    // gemm with normal-layout mat * normal-layout mat; also consumes operands ownerships
    pub fn gemm_owned(&self, lhs: BlasTensor, rhs: BlasTensor) -> BlasTensor {
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
    pub fn gemm_side_effect(&self, lhs: &BlasTensor, rhs: &BlasTensor, out: &mut BlasTensor) -> () {
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
    fn test_gemm_owned() {
        let a = BlasTensor::ones(vec![17, 23]);
        let b = BlasTensor::ones(vec![23, 18]);
        let cref = BlasTensor::from_vec_shape([23.0; 17 * 18].to_vec(), vec![17, 18]);

        let exec = BlasExecutor::new();
        let c = exec.gemm_owned(a, b);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [17, 18]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_gemm_side_effect() {
        let a = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let b = BlasTensor::ones(vec![3, 2]);
        let mut c = BlasTensor::zeros(vec![2, 2]);
        let cref = BlasTensor::from_vec_shape(vec![6.6000004, 6.6000004, 16.5, 16.5], vec![2, 2]);

        let exec = BlasExecutor::new();
        exec.gemm_side_effect(&a, &b, &mut c);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 2]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_addf32_owned() {
        let a = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let b = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let cref = BlasTensor::from_vec_shape(vec![2.2, 4.4, 6.6, 8.8, 11., 13.2], vec![2, 3]);

        let exec = BlasExecutor::new();
        let c = exec.addf32_owned(a, b);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_subf32_owned() {
        let a = BlasTensor::from_vec_shape(vec![2.2, 4.4, 6.6, 8.8, 11., 13.2], vec![2, 3]);
        let b = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let cref = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);

        let exec = BlasExecutor::new();
        let c = exec.subf32_owned(a, b);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_mulf32_owned() {
        let a = BlasTensor::from_vec_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = BlasTensor::from_vec_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let cref = BlasTensor::from_vec_shape(vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0], vec![2, 3]);

        let exec = BlasExecutor::new();
        let c = exec.mulf32_owned(a, b);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_divf32_owned() {
        let a = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let b = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let cref = BlasTensor::from_vec_shape(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![2, 3]);

        let exec = BlasExecutor::new();
        let c = exec.divf32_owned(a, b);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_addf32_side_effect() {
        let a = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let b = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let mut c = BlasTensor::zeros(vec![2, 3]);
        let cref = BlasTensor::from_vec_shape(vec![2.2, 4.4, 6.6, 8.8, 11., 13.2], vec![2, 3]);

        let exec = BlasExecutor::new();
        exec.addf32_side_effect(&a, &b, &mut c);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_subf32_side_effect() {
        let a = BlasTensor::from_vec_shape(vec![2.2, 4.4, 6.6, 8.8, 11., 13.2], vec![2, 3]);
        let b = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let mut c = BlasTensor::zeros(vec![2, 3]);
        let cref = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);

        let exec = BlasExecutor::new();
        exec.subf32_side_effect(&a, &b, &mut c);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_mulf32_side_effect() {
        let a = BlasTensor::from_vec_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = BlasTensor::from_vec_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let mut c = BlasTensor::zeros(vec![2, 3]);
        let cref = BlasTensor::from_vec_shape(vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0], vec![2, 3]);

        let exec = BlasExecutor::new();
        exec.mulf32_side_effect(&a, &b, &mut c);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_divf32_side_effect() {
        let a = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let b = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let mut c = BlasTensor::zeros(vec![2, 3]);
        let cref = BlasTensor::from_vec_shape(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![2, 3]);

        let exec = BlasExecutor::new();
        exec.divf32_side_effect(&a, &b, &mut c);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_binary_compute_addf_owned() {
        let a = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let b = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let cref = BlasTensor::from_vec_shape(vec![2.2, 4.4, 6.6, 8.8, 11., 13.2], vec![2, 3]);
        let op = BlasOpCode::AddF;

        let exec = BlasExecutor::new();
        let c = exec.binary_compute_owned(op, a, b);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_binary_compute_subf_owned() {
        let a = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let b = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let cref = BlasTensor::from_vec_shape(vec![0.0; 6], vec![2, 3]);
        let op = BlasOpCode::SubF;

        let exec = BlasExecutor::new();
        let c = exec.binary_compute_owned(op, a, b);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_binary_compute_divf_owned() {
        let a = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let b = BlasTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
        let cref = BlasTensor::from_vec_shape(vec![1.0; 6], vec![2, 3]);
        let op = BlasOpCode::DivF;

        let exec = BlasExecutor::new();
        let c = exec.binary_compute_owned(op, a, b);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_binary_compute_gemm_owned() {
        let a = BlasTensor::ones(vec![17, 23]);
        let b = BlasTensor::ones(vec![23, 18]);
        let cref = BlasTensor::from_vec_shape([23.0; 17 * 18].to_vec(), vec![17, 18]);
        let op = BlasOpCode::GemmF;

        let exec = BlasExecutor::new();
        let c = exec.binary_compute_owned(op, a, b);
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [17, 18]);
        assert_eq!(c, cref);
    }
}
