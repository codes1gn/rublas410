use ndarray::prelude::*;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::Distribution;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::rand_distr::Uniform;
use num_traits::Zero;
use std::fmt::Debug;

use ndarray::prelude::*;
use ndarray::Array;
use ndarray::*;
pub use ndarray_rand::RandomExt;
use ndarray_rand::F32;

// TODO consider hide TensorKind, and expose a into_raw_vec for BlasTensor
#[derive(Debug, PartialEq)]
pub enum TensorKind {
    FloatVector(Array1<f32>),
    FloatMatrix(Array2<f32>),
    DoubleVector(Array1<f64>),
    DoubleMatrix(Array2<f64>),
    Int32Vector(Array1<i32>),
    Int32Matrix(Array2<i32>),
    Int8Vector(Array1<i8>),
    Int8Matrix(Array2<i8>),
}

impl From<Array1<f32>> for TensorKind {
    fn from(who: Array1<f32>) -> Self {
        TensorKind::FloatVector(who)
    }
}

impl From<Array2<f32>> for TensorKind {
    fn from(who: Array2<f32>) -> Self {
        TensorKind::FloatMatrix(who)
    }
}

impl From<Array1<f64>> for TensorKind {
    fn from(who: Array1<f64>) -> Self {
        TensorKind::DoubleVector(who)
    }
}

impl From<Array2<f64>> for TensorKind {
    fn from(who: Array2<f64>) -> Self {
        TensorKind::DoubleMatrix(who)
    }
}

impl From<Array1<i32>> for TensorKind {
    fn from(who: Array1<i32>) -> Self {
        TensorKind::Int32Vector(who)
    }
}

impl From<Array2<i32>> for TensorKind {
    fn from(who: Array2<i32>) -> Self {
        TensorKind::Int32Matrix(who)
    }
}

// Purpose of this wrapper layer:
// 1. simplify usage, hide generics configuration
// 2. directly support high-order tensor but use <2D arrays for performance
// 3. hide implementation of strides/paddings in this layer and make this controllable
// 4. expose fixed pattern to CRT, act like ISA style
//
//
// TODO only use Array1, Array2 as the data field
// to fit blas config
// TODO maybe need a dedicated Shape/Dimension struct
// TODO only support D1, D2, D3, D4 now
// TODO only allow channel_last data layout that contineous along
// last dims
#[derive(Debug, PartialEq)]
pub struct BlasTensor {
    pub data: TensorKind,
    pub shape: Vec<usize>,
}

impl BlasTensor {
    pub fn ndims(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn from_vec(raw_data: Vec<f32>) -> BlasTensor {
        let raw_shape = vec![raw_data.len()];
        BlasTensor {
            data: TensorKind::from(Array::from(raw_data)),
            shape: raw_shape,
        }
    }

    pub fn from_vec_shape(raw_data: Vec<f32>, shape: Vec<usize>) -> BlasTensor {
        let dims = shape.len();
        if dims == 1 {
            return BlasTensor {
                data: TensorKind::from(
                    Array1::<f32>::from_shape_vec([shape[0]], raw_data).unwrap(),
                ),
                shape: shape,
            };
        } else if dims == 2 {
            return Self {
                data: TensorKind::from(
                    Array2::<f32>::from_shape_vec([shape[0], shape[1]], raw_data).unwrap(),
                ),
                shape: shape,
            };
        } else if dims == 3 {
            return Self {
                data: TensorKind::from(
                    Array2::<f32>::from_shape_vec([shape[0] * shape[1], shape[2]], raw_data)
                        .unwrap(),
                ),
                shape: shape,
            };
        } else if dims == 4 {
            return Self {
                data: TensorKind::from(
                    Array2::<f32>::from_shape_vec(
                        [shape[0] * shape[1] * shape[2], shape[3]],
                        raw_data,
                    )
                    .unwrap(),
                ),
                shape: shape,
            };
        } else {
            panic!("not support float tensor with 5 dims or more");
        }
    }

    // TODO generic function
    pub fn from_vec_shape_i32(raw_data: Vec<i32>, shape: Vec<usize>) -> BlasTensor {
        let dims = shape.len();
        if dims == 1 {
            return BlasTensor {
                data: TensorKind::from(
                    Array1::<i32>::from_shape_vec([shape[0]], raw_data).unwrap(),
                ),
                shape: shape,
            };
        } else if dims == 2 {
            return Self {
                data: TensorKind::from(
                    Array2::<i32>::from_shape_vec([shape[0], shape[1]], raw_data).unwrap(),
                ),
                shape: shape,
            };
        } else if dims == 3 {
            return Self {
                data: TensorKind::from(
                    Array2::<i32>::from_shape_vec([shape[0] * shape[1], shape[2]], raw_data)
                        .unwrap(),
                ),
                shape: shape,
            };
        } else if dims == 4 {
            return Self {
                data: TensorKind::from(
                    Array2::<i32>::from_shape_vec(
                        [shape[0] * shape[1] * shape[2], shape[3]],
                        raw_data,
                    )
                    .unwrap(),
                ),
                shape: shape,
            };
        } else {
            panic!("not support float tensor with 5 dims or more");
        }
    }

    // TODO patch match on shape len
    pub fn zeros(shape: Vec<usize>) -> BlasTensor {
        let dims = shape.len();
        if dims == 1 {
            return Self {
                data: TensorKind::from(Array1::<f32>::zeros([shape[0]])),
                shape: shape,
            };
        } else if dims == 2 {
            return Self {
                data: TensorKind::from(Array2::<f32>::zeros([shape[0], shape[1]])),
                shape: shape,
            };
        } else if dims == 3 {
            return Self {
                data: TensorKind::from(Array2::<f32>::zeros([shape[0] * shape[1], shape[2]])),
                shape: shape,
            };
        } else if dims == 4 {
            return Self {
                data: TensorKind::from(Array2::<f32>::zeros([
                    shape[0] * shape[1] * shape[2],
                    shape[3],
                ])),
                shape: shape,
            };
        } else {
            panic!("not support float tensor with 5 dims or more");
        }
    }

    pub fn ones(shape: Vec<usize>) -> BlasTensor {
        let dims = shape.len();
        if dims == 1 {
            return Self {
                data: TensorKind::from(Array1::<f32>::ones([shape[0]])),
                shape: shape,
            };
        } else if dims == 2 {
            return Self {
                data: TensorKind::from(Array2::<f32>::ones([shape[0], shape[1]])),
                shape: shape,
            };
        } else if dims == 3 {
            return Self {
                data: TensorKind::from(Array2::<f32>::ones([shape[0] * shape[1], shape[2]])),
                shape: shape,
            };
        } else if dims == 4 {
            return Self {
                data: TensorKind::from(Array2::<f32>::ones([
                    shape[0] * shape[1] * shape[2],
                    shape[3],
                ])),
                shape: shape,
            };
        } else {
            panic!("not support float tensor with 5 dims or more");
        }
    }

    pub fn zeros_double(shape: Vec<usize>) -> BlasTensor {
        let dims = shape.len();
        if dims == 1 {
            return Self {
                data: TensorKind::from(Array1::<f64>::zeros([shape[0]])),
                shape: shape,
            };
        } else if dims == 2 {
            return Self {
                data: TensorKind::from(Array2::<f64>::zeros([shape[0], shape[1]])),
                shape: shape,
            };
        } else if dims == 3 {
            return Self {
                data: TensorKind::from(Array2::<f64>::zeros([shape[0] * shape[1], shape[2]])),
                shape: shape,
            };
        } else if dims == 4 {
            return Self {
                data: TensorKind::from(Array2::<f64>::zeros([
                    shape[0] * shape[1] * shape[2],
                    shape[3],
                ])),
                shape: shape,
            };
        } else {
            panic!("not support float tensor with 5 dims or more");
        }
    }

    // TODO refactor to pattern-match style
    pub fn uniform(shape: Vec<usize>, min: f32, max: f32) -> BlasTensor {
        let dims = shape.len();
        if dims == 1 {
            return Self {
                data: TensorKind::from(Array1::<f32>::random(
                    shape[0],
                    Uniform::<f32>::new(min, max),
                )),
                shape: shape,
            };
        } else if dims == 2 {
            return Self {
                data: TensorKind::from(Array2::<f32>::random(
                    [shape[0], shape[1]],
                    Uniform::<f32>::new(min, max),
                )),
                shape: shape,
            };
        } else if dims == 3 {
            return Self {
                data: TensorKind::from(Array2::<f32>::random(
                    [shape[0] * shape[1], shape[2]],
                    Uniform::<f32>::new(min, max),
                )),
                shape: shape,
            };
        } else if dims == 4 {
            return Self {
                data: TensorKind::from(Array2::<f32>::random(
                    [shape[0] * shape[1] * shape[2], shape[3]],
                    Uniform::<f32>::new(min, max),
                )),
                shape: shape,
            };
        } else {
            panic!("not support tensor with 5 dims or more");
        }
    }

    pub fn uniform_double(shape: Vec<usize>, min: f64, max: f64) -> BlasTensor {
        let dims = shape.len();
        if dims == 1 {
            return Self {
                data: TensorKind::from(Array1::<f64>::random(
                    shape[0],
                    Uniform::<f64>::new(min, max),
                )),
                shape: shape,
            };
        } else if dims == 2 {
            return Self {
                data: TensorKind::from(Array2::<f64>::random(
                    [shape[0], shape[1]],
                    Uniform::<f64>::new(min, max),
                )),
                shape: shape,
            };
        } else if dims == 3 {
            return Self {
                data: TensorKind::from(Array2::<f64>::random(
                    [shape[0] * shape[1], shape[2]],
                    Uniform::<f64>::new(min, max),
                )),
                shape: shape,
            };
        } else if dims == 4 {
            return Self {
                data: TensorKind::from(Array2::<f64>::random(
                    [shape[0] * shape[1] * shape[2], shape[3]],
                    Uniform::<f64>::new(min, max),
                )),
                shape: shape,
            };
        } else {
            panic!("not support tensor with 5 dims or more");
        }
    }

    pub fn normal(shape: Vec<usize>, mean: f32, std: f32) -> BlasTensor {
        let dims = shape.len();
        if dims == 1 {
            return Self {
                data: TensorKind::from(Array1::<f32>::random(
                    shape[0],
                    Normal::<f32>::new(mean, std).unwrap(),
                )),
                shape: shape,
            };
        } else if dims == 2 {
            return Self {
                data: TensorKind::from(Array2::<f32>::random(
                    [shape[0], shape[1]],
                    Normal::<f32>::new(mean, std).unwrap(),
                )),
                shape: shape,
            };
        } else if dims == 3 {
            return Self {
                data: TensorKind::from(Array2::<f32>::random(
                    [shape[0] * shape[1], shape[2]],
                    Normal::<f32>::new(mean, std).unwrap(),
                )),
                shape: shape,
            };
        } else if dims == 4 {
            return Self {
                data: TensorKind::from(Array2::<f32>::random(
                    [shape[0] * shape[1] * shape[2], shape[3]],
                    Normal::<f32>::new(mean, std).unwrap(),
                )),
                shape: shape,
            };
        } else {
            panic!("not support tensor with 5 dims or more");
        }
    }

    pub fn normal_double(shape: Vec<usize>, mean: f64, std: f64) -> BlasTensor {
        let dims = shape.len();
        if dims == 1 {
            return Self {
                data: TensorKind::from(Array1::<f64>::random(
                    shape[0],
                    Normal::<f64>::new(mean, std).unwrap(),
                )),
                shape: shape,
            };
        } else if dims == 2 {
            return Self {
                data: TensorKind::from(Array2::<f64>::random(
                    [shape[0], shape[1]],
                    Normal::<f64>::new(mean, std).unwrap(),
                )),
                shape: shape,
            };
        } else if dims == 3 {
            return Self {
                data: TensorKind::from(Array2::<f64>::random(
                    [shape[0] * shape[1], shape[2]],
                    Normal::<f64>::new(mean, std).unwrap(),
                )),
                shape: shape,
            };
        } else if dims == 4 {
            return Self {
                data: TensorKind::from(Array2::<f64>::random(
                    [shape[0] * shape[1] * shape[2], shape[3]],
                    Normal::<f64>::new(mean, std).unwrap(),
                )),
                shape: shape,
            };
        } else {
            panic!("not support tensor with 5 dims or more");
        }
    }
}

#[cfg(test)]

mod tests {
    use super::*;
    use ndarray::*;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_zeros_1d() {
        let blast = BlasTensor::zeros(vec![64]);
        let reft = TensorKind::FloatVector(Array::<f32, _>::zeros(64));
        assert_eq!(blast.data, reft);
    }

    #[test]
    fn test_zeros_1d_double() {
        let blast = BlasTensor::zeros_double(vec![64]);
        let reft = TensorKind::DoubleVector(Array::<f64, _>::zeros(64));
        assert_eq!(blast.data, reft);
    }

    // TODO support i-types
    // #[test]
    // fn test_zeros_1d_i8() {
    //     let blast = BlasTensor::<i8>::zeros(vec![64]);
    //     let reft = TensorKind::Vector(Array::<i8, _>::zeros(64));
    //     assert_eq!(blast.data, reft);
    // }

    #[test]
    fn test_zeros_2d() {
        let blast = BlasTensor::zeros(vec![64, 32]);
        let reft = TensorKind::FloatMatrix(Array::<f32, _>::zeros((64, 32)));
        assert_eq!(blast.data, reft);
    }

    #[test]
    fn test_ones_2d() {
        let blast = BlasTensor::ones(vec![64, 32]);
        let reft = TensorKind::FloatMatrix(Array::<f32, _>::ones((64, 32)));
        assert_eq!(blast.data, reft);
    }

    #[test]
    fn test_zeros_3d() {
        let blast = BlasTensor::zeros(vec![8, 64, 32]);
        let reft = TensorKind::FloatMatrix(Array::<f32, _>::zeros((512, 32)));
        assert_eq!(blast.data, reft);
    }

    #[test]
    fn test_zeros_4d() {
        let blast = BlasTensor::zeros(vec![8, 2, 64, 32]);
        let reft = TensorKind::FloatMatrix(Array::<f32, _>::zeros((1024, 32)));
        assert_eq!(blast.data, reft);
    }

    #[test]
    fn test_ones_4d() {
        let blast = BlasTensor::ones(vec![8, 2, 64, 32]);
        let reft = TensorKind::FloatMatrix(Array::<f32, _>::ones((1024, 32)));
        assert_eq!(blast.data, reft);
    }

    // TODO impl rand_isaac with fixed seed, then compare contents
    #[test]
    fn test_uniform_1d() {
        let blast = BlasTensor::uniform(vec![64], -1.0, 1.0);
        let reft = TensorKind::FloatVector(Array::random(64, Uniform::new(-1f32, 1.)));
        // assert_eq!(blast.data.into().shape(), reft.into().shape());
    }

    #[test]
    fn test_uniform_double_1d() {
        let blast = BlasTensor::uniform(vec![64], -1f32, 1.0);
        let reft = TensorKind::DoubleVector(Array::random(64, Uniform::new(-1f64, 1.)));
        // assert_eq!(blast.data, reft);
    }

    #[test]
    fn test_uniform_2d() {
        let blast = BlasTensor::uniform(vec![64, 32], -1f32, 1.0);
        let reft = TensorKind::FloatMatrix(Array::random([64, 32], Uniform::new(-1f32, 1.)));
        // assert_eq!(blast.data, reft);
    }

    #[test]
    fn test_normal_2d() {
        let blast = BlasTensor::normal(vec![64, 32], 0.0f32, 1.0);
        let reft =
            TensorKind::FloatMatrix(Array::random([64, 32], Normal::new(0.0f32, 1.).unwrap()));
        // assert_eq!(blast.data, reft);
    }

    #[test]
    fn test_normal_double_2d() {
        let blast = BlasTensor::uniform_double(vec![64, 32], -1f64, 1.0);
        let reft =
            TensorKind::DoubleMatrix(Array::random([64, 32], Normal::new(-1f64, 1.).unwrap()));
        // assert_eq!(blast.data, reft);
    }

    #[test]
    fn test_uniform_double_2d() {
        let blast = BlasTensor::uniform_double(vec![64, 32], -1f64, 1.0);
        let reft = TensorKind::DoubleMatrix(Array::random([64, 32], Uniform::new(-1f64, 1.)));
        // assert_eq!(blast.data, reft);
    }

    #[test]
    fn test_uniform_3d() {
        let blast = BlasTensor::uniform(vec![8, 64, 32], -1f32, 1.0);
        let reft = TensorKind::FloatMatrix(Array::random([512, 32], Uniform::new(-1f32, 1.)));
        // assert_eq!(blast.data, reft);
    }

    #[test]
    fn test_uniform_4d() {
        let blast = BlasTensor::uniform(vec![8, 2, 64, 32], -1f32, 1.0);
        let reft = TensorKind::FloatMatrix(Array::random([1024, 32], Uniform::new(-1f32, 1.)));
        // assert_eq!(blast.data, reft);
    }

    #[test]
    fn test_build_from_1d() {
        let blast = BlasTensor::from_vec(vec![1.7, 2.3, 3.3, 4.1]);
        let reft = TensorKind::FloatVector(Array::from(vec![1.7, 2.3, 3.3, 4.1]));
        assert_eq!(blast.shape(), vec![4]);
        assert_eq!(blast.data, reft);
    }

    #[test]
    fn test_build_from_2d() {
        let blast =
            BlasTensor::from_vec_shape(vec![1.7, 2.3, 3.3, 4.1, 1.7, 2.3, 3.3, 4.1], vec![2, 4]);
        let reft = TensorKind::FloatMatrix(
            Array::from_shape_vec([2, 4], vec![1.7, 2.3, 3.3, 4.1, 1.7, 2.3, 3.3, 4.1]).unwrap(),
        );
        assert_eq!(blast.shape(), vec![2, 4]);
        assert_eq!(blast.data, reft);
    }

    #[test]
    fn test_build_from_2d_i32() {
        let blast = BlasTensor::from_vec_shape_i32(vec![1i32, 2, 3, 4, 7, 2, 3, 4], vec![2, 4]);
        let reft = TensorKind::Int32Matrix(
            Array::from_shape_vec([2, 4], vec![1i32, 2, 3, 4, 7, 2, 3, 4]).unwrap(),
        );
        assert_eq!(blast.shape(), vec![2, 4]);
        assert_eq!(blast.data, reft);
    }
}
