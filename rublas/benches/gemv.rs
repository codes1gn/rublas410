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

// TODO add a gemv_legacy
fn gemv_zero(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("gemv_zero");
    for Msize in vec![16, 64, 256, 1024, 4096].iter() {
        for Ksize in vec![16, 64, 256, 1024, 4096].iter() {
            bench_group.bench_with_input(
                BenchmarkId::new(format!("M{}_K{}_f32", Msize, Ksize), 0),
                Msize,
                |bench, msize| {
                    let a = Array2::<f32>::zeros((*Msize, *Ksize));
                    let (m, n) = a.dim();
                    let x = Array1::<f32>::zeros(n);
                    let mut y = Array1::<f32>::zeros(m);
                    bench.iter(|| {
                        black_box(general_mat_vec_mul(1.0, &a, &x, 1.0, &mut y));
                    });
                },
            );
            bench_group.bench_with_input(
                BenchmarkId::new(format!("M{}_K{}_f64", Msize, Ksize), 0),
                Msize,
                |bench, msize| {
                    let a = Array2::<f64>::zeros((*Msize, *Ksize));
                    let (m, n) = a.dim();
                    let x = Array1::<f64>::zeros(n);
                    let mut y = Array1::<f64>::zeros(m);
                    bench.iter(|| {
                        black_box(general_mat_vec_mul(1.0, &a, &x, 1.0, &mut y));
                    });
                },
            );
        }
    }
}

fn gemv_uniform(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("gemv_uniform");
    for Msize in vec![16, 64, 256, 1024, 4096].iter() {
        for Ksize in vec![16, 64, 256, 1024, 4096].iter() {
            bench_group.bench_with_input(
                BenchmarkId::new(format!("M{}_K{}_f32", Msize, Ksize), 0),
                Msize,
                |bench, msize| {
                    let a = Array2::<f32>::random((*Msize, *Ksize), Uniform::new(-1f32, 1.0));
                    let (m, n) = a.dim();
                    let x = Array1::<f32>::random(n, Uniform::new(-1f32, 1.0));
                    let mut y = Array1::<f32>::random(m, Uniform::new(-1f32, 1.0));
                    bench.iter(|| {
                        black_box(general_mat_vec_mul(1.0, &a, &x, 1.0, &mut y));
                    });
                },
            );
            bench_group.bench_with_input(
                BenchmarkId::new(format!("M{}_K{}_f64", Msize, Ksize), 0),
                Msize,
                |bench, msize| {
                    let a = Array2::<f64>::random((*Msize, *Ksize), Uniform::new(-1f64, 1.0));
                    let (m, n) = a.dim();
                    let x = Array1::<f64>::random(n, Uniform::new(-1f64, 1.0));
                    let mut y = Array1::<f64>::random(m, Uniform::new(-1f64, 1.0));
                    bench.iter(|| {
                        black_box(general_mat_vec_mul(1.0, &a, &x, 1.0, &mut y));
                    });
                },
            );
        }
    }
}

criterion_group!(gemv_tests, gemv_zero);
criterion_main!(gemv_tests);

// #[bench]
// fn gemv_64_64f(bench: &mut Bencher) {
//     let a = Array::zeros((64, 64).f());
//     let (m, n) = a.dim();
//     let x = Array::zeros(n);
//     let mut y = Array::zeros(m);
//     bench.iter(|| {
//         general_mat_vec_mul(1.0, &a, &x, 1.0, &mut y);
//     });
// }
//
// #[bench]
// fn gemv_64_32(bench: &mut Bencher) {
//     let a = Array::zeros((64, 32));
//     let (m, n) = a.dim();
//     let x = Array::zeros(n);
//     let mut y = Array::zeros(m);
//     bench.iter(|| {
//         general_mat_vec_mul(1.0, &a, &x, 1.0, &mut y);
//     });
// }
