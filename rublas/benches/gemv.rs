use criterion::*;
use ndarray::*;
use ndarray::Array;
use ndarray::prelude::*;
use ndarray::linalg::general_mat_vec_mul;

fn gemv_64_64c(crit: &mut Criterion) {
    let mut bench_group = crit.benchmark_group("gemm_test");
    let mat_size = 64;
    bench_group.bench_with_input(BenchmarkId::new("gemv_64_64c/1", mat_size), &mat_size, |bench, msize| {
        let a = Array2::<f32>::zeros((*msize, *msize));
        let (m, n) = a.dim();
        let x = Array1::<f32>::zeros(n);
        let mut y = Array1::<f32>::zeros(m);
        bench.iter(|| {
            general_mat_vec_mul(1.0, &a, &x, 1.0, &mut y);
        });
    
    });
    
}
// fn eig_small(c: &mut Criterion) {
//     let mut group = c.benchmark_group("eig");
//     for &n in &[4, 8, 16, 32, 64, 128] {
//         group.bench_with_input(BenchmarkId::new("vecs/C", n), &n, |b, n| {
//             let a: Array2<f64> = random((*n, *n));
//             b.iter(|| {
//                 let (_e, _vecs) = a.eig().unwrap();
//             })
//         });
//         group.bench_with_input(BenchmarkId::new("vecs/F", n), &n, |b, n| {
//             let a: Array2<f64> = random((*n, *n).f());
//             b.iter(|| {
//                 let (_e, _vecs) = a.eig().unwrap();
//             })
//         });
//         group.bench_with_input(BenchmarkId::new("vals/C", n), &n, |b, n| {
//             let a: Array2<f64> = random((*n, *n));
//             b.iter(|| {
//                 let _result = a.eigvals().unwrap();
//             })
//         });
//         group.bench_with_input(BenchmarkId::new("vals/F", n), &n, |b, n| {
//             let a: Array2<f64> = random((*n, *n).f());
//             b.iter(|| {
//                 let _result = a.eigvals().unwrap();
//             })
//         });
//     }
// }

criterion_group!(gemm_test, gemv_64_64c);
criterion_main!(gemm_test);



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
