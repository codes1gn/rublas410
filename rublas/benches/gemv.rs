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

use criterion::*;
use rublas::prelude::*;

fn small_gemv(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("gemv");
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

criterion_group!(gemv_tests, small_gemv);
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
