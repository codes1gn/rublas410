#[cfg(feature = "openblas")]
extern crate ndarray_blas as ndarray;

#[cfg(feature = "netlib")]
extern crate ndarray_blas as ndarray;

#[cfg(all(
    feature = "default",
    not(feature = "openblas"),
    not(feature = "netlib")
))]
extern crate ndarray_vanilla as ndarray;

use ndarray::linalg::general_mat_vec_mul;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_rand::F32;

use criterion::*;
use rublas::prelude::*;

fn zeros_builder(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("zeros_builder");
    for mat_size in vec![4096].iter() {
        bench_group.bench_with_input(
            BenchmarkId::new(format!("Array2"), mat_size),
            mat_size,
            |bench, msize| {
                bench.iter(|| {
                    black_box(Array::<f32, _>::zeros((*msize, *msize)));
                });
            },
        );
        bench_group.bench_with_input(
            BenchmarkId::new(format!("BlasTensor"), mat_size),
            mat_size,
            |bench, msize| {
                bench.iter(|| {
                    black_box(BlasTensor::zeros(vec![*msize, *msize]));
                });
            },
        );
    }
}

fn uniform_builder(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("uniform_builder");
    for mat_size in vec![4096].iter() {
        bench_group.bench_with_input(
            BenchmarkId::new(format!("Array2"), mat_size),
            mat_size,
            |bench, msize| {
                bench.iter(|| {
                    black_box(Array::random((*msize, *msize), Uniform::new(-1f32, 1.)));
                });
            },
        );
        bench_group.bench_with_input(
            BenchmarkId::new(format!("BlasTensor"), mat_size),
            mat_size,
            |bench, msize| {
                bench.iter(|| {
                    black_box(BlasTensor::uniform(vec![*msize, *msize], -1f32, 1.));
                });
            },
        );
    }
}

// TODO other rng patterns
criterion_group!(tensor_create_ablation_tests, zeros_builder, uniform_builder);
criterion_main!(tensor_create_ablation_tests);
