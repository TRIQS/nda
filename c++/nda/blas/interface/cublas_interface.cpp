// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Implementation details for blas/interface/cublas_interface.hpp.
 */

#include "./cublas_interface.hpp"
#include "../tools.hpp"
#include "../../device.hpp"
#include "../../exceptions.hpp"
#include "../../traits.hpp"

#ifdef NDA_HAVE_MAGMA
#include "magma_v2.h"

#include <exception>
#endif

#include <vector>
#include <type_traits>

namespace nda::blas::device {

  // Local function to get unique CuBlas handle.
  inline cublasHandle_t &get_handle() {
    struct handle_storage_t { // RAII for the handle
      handle_storage_t() { cublasCreate(&handle); }
      ~handle_storage_t() { cublasDestroy(handle); }
      cublasHandle_t handle = {};
    };
    static auto sto = handle_storage_t{};
    return sto.handle;
  }

#ifdef NDA_HAVE_MAGMA
  // Local function to get Magma op.
  constexpr magma_trans_t get_magma_op(char op) {
    switch (op) {
      case 'N': return MagmaNoTrans; break;
      case 'T': return MagmaTrans; break;
      case 'C': return MagmaConjTrans; break;
      default: std::terminate(); return {};
    }
  }

  // Local function to get Magma queue.
  auto &get_magma_queue() {
    struct queue_t {
      queue_t() {
        int device{};
        magma_getdevice(&device);
        magma_queue_create(device, &q);
      }
      ~queue_t() { magma_queue_destroy(q); }
      operator magma_queue_t() { return q; }

      private:
      magma_queue_t q = {};
    };
    static queue_t q = {};
    return q;
  }
#endif

  // Per-thread option to turn on/off the cudaDeviceSynchronize after cublas library calls.
  thread_local bool synchronize = true; // NOLINT (per-thread option is on purpose)
  void set_synchronization(bool do_sync) noexcept { synchronize = do_sync; }
  bool get_synchronization() noexcept { return synchronize; }

// Macro to check cublas calls.
#define CUBLAS_CHECK(X, ...)                                                                                                                         \
  {                                                                                                                                                  \
    auto err = X(get_handle(), __VA_ARGS__);                                                                                                         \
    if (err != CUBLAS_STATUS_SUCCESS) {                                                                                                              \
      NDA_RUNTIME_ERROR << AS_STRING(X) << " failed \n"                                                                                              \
                        << " cublasGetStatusName: " << cublasGetStatusName(err) << "\n"                                                              \
                        << " cublasGetStatusString: " << cublasGetStatusString(err) << "\n";                                                         \
    }                                                                                                                                                \
    cuda_device_sync(synchronize, AS_STRING(X));                                                                                                     \
  }

  // Anonymous namespace for some file local helper functions.
  namespace {

    // Cuda data type conversion.
    template <typename T>
    constexpr auto cuda_data_type() {
      if constexpr (std::is_same_v<T, float>) {
        return CUDA_R_32F;
      } else if constexpr (std::is_same_v<T, double>) {
        return CUDA_R_64F;
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return CUDA_C_32F;
      } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return CUDA_C_64F;
      }
    }

    // Cuda compute type conversion.
    template <typename T>
    constexpr auto cuda_compute_type() {
      if constexpr (std::is_same_v<T, float> or std::is_same_v<T, std::complex<float>>) {
        return CUBLAS_COMPUTE_32F;
      } else if constexpr (std::is_same_v<T, double> or std::is_same_v<T, std::complex<double>>) {
        return CUBLAS_COMPUTE_64F;
      }
    }

    // Helper function to call CUDA's cublasGemmGroupedBatchedEx routine.
    template <typename T>
    void cuda_gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, T alpha, const T **a, int *lda, const T **b, int *ldb, T beta, T **c,
                          int *ldc, int batch_count) {
      auto data_t    = cuda_data_type<T>();
      auto compute_t = cuda_compute_type<T>();
      auto vec_op_a  = std::vector<cublasOperation_t>(batch_count, get_cublas_op(op_a));
      auto vec_op_b  = std::vector<cublasOperation_t>(batch_count, get_cublas_op(op_b));
      auto vec_alpha = std::vector<T>(batch_count, alpha);
      auto vec_beta  = std::vector<T>(batch_count, beta);
      auto vec_sizes = std::vector<int>(batch_count, 1);
      CUBLAS_CHECK(cublasGemmGroupedBatchedEx, vec_op_a.data(), vec_op_b.data(), m, n, k, vec_alpha.data(), (const void **)a, data_t, lda,
                   (const void **)b, data_t, ldb, vec_beta.data(), (void **)c, data_t, ldc, batch_count, vec_sizes.data(), compute_t);
    }

    // Helper function to call Magma's magma_gemm_vbatched routine.
#ifdef NDA_HAVE_MAGMA
    template <typename T>
    void magma_gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, T alpha, const T **a, int *lda, const T **b, int *ldb, T beta, T **c,
                           int *ldc, int batch_count) {
      if constexpr (std::is_same_v<T, std::complex<float>>) {
        magmablas_cgemm_vbatched(get_magma_op(op_a), get_magma_op(op_b), m, n, k, cucplx(alpha), cucplx(a), lda, cucplx(b), ldb, cucplx(beta),
                                 cucplx(c), ldc, batch_count, get_magma_queue());
      } else {
        magmablas_zgemm_vbatched(get_magma_op(op_a), get_magma_op(op_b), m, n, k, cucplx(alpha), cucplx(a), lda, cucplx(b), ldb, cucplx(beta),
                                 cucplx(c), ldc, batch_count, get_magma_queue());
      }
      if (synchronize) magma_queue_sync(get_magma_queue());
      cuda_device_sync(synchronize, "magma_gemm_vbatch");
    }
#else
    template <typename T>
    void magma_gemm_vbatch(char, char, int *, int *, int *, T, const T **, int *, const T **, int *, T, T **, int *, int) {
      NDA_RUNTIME_ERROR << "nda::blas::device::gemmv_batch with complex types requires Magma. Configure nda with -DMagmaSupport=ON";
    }
#endif

  } // namespace

  // axpy
  void axpy(int n, float alpha, const float *x, int incx, float *y, int incy) { CUBLAS_CHECK(cublasSaxpy, n, &alpha, x, incx, y, incy); }
  void axpy(int n, std::complex<float> alpha, const std::complex<float> *x, int incx, std::complex<float> *y, int incy) {
    CUBLAS_CHECK(cublasCaxpy, n, cuscalar(alpha), cucplx(x), incx, cucplx(y), incy);
  }
  void axpy(int n, double alpha, const double *x, int incx, double *y, int incy) { CUBLAS_CHECK(cublasDaxpy, n, &alpha, x, incx, y, incy); }
  void axpy(int n, std::complex<double> alpha, const std::complex<double> *x, int incx, std::complex<double> *y, int incy) {
    CUBLAS_CHECK(cublasZaxpy, n, cuscalar(alpha), cucplx(x), incx, cucplx(y), incy);
  }

  // copy
  void copy(int n, const float *x, int incx, float *y, int incy) { CUBLAS_CHECK(cublasScopy, n, x, incx, y, incy); }
  void copy(int n, const std::complex<float> *x, int incx, std::complex<float> *y, int incy) {
    CUBLAS_CHECK(cublasCcopy, n, cucplx(x), incx, cucplx(y), incy);
  }
  void copy(int n, const double *x, int incx, double *y, int incy) { CUBLAS_CHECK(cublasDcopy, n, x, incx, y, incy); }
  void copy(int n, const std::complex<double> *x, int incx, std::complex<double> *y, int incy) {
    CUBLAS_CHECK(cublasZcopy, n, cucplx(x), incx, cucplx(y), incy);
  }

  // dot and dotc
  float dot(int m, const float *x, int incx, const float *y, int incy) {
    float res{};
    CUBLAS_CHECK(cublasSdot, m, x, incx, y, incy, &res);
    return res;
  }
  std::complex<float> dot(int m, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy) {
    cuComplex res;
    CUBLAS_CHECK(cublasCdotu, m, cucplx(x), incx, cucplx(y), incy, &res);
    return {res.x, res.y};
  }
  std::complex<float> dotc(int m, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy) {
    cuComplex res;
    CUBLAS_CHECK(cublasCdotc, m, cucplx(x), incx, cucplx(y), incy, &res);
    return {res.x, res.y};
  }
  double dot(int m, const double *x, int incx, const double *y, int incy) {
    double res{};
    CUBLAS_CHECK(cublasDdot, m, x, incx, y, incy, &res);
    return res;
  }
  std::complex<double> dot(int m, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy) {
    cuDoubleComplex res;
    CUBLAS_CHECK(cublasZdotu, m, cucplx(x), incx, cucplx(y), incy, &res);
    return {res.x, res.y};
  }
  std::complex<double> dotc(int m, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy) {
    cuDoubleComplex res;
    CUBLAS_CHECK(cublasZdotc, m, cucplx(x), incx, cucplx(y), incy, &res);
    return {res.x, res.y};
  }

  // gemm
  void gemm(char op_a, char op_b, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    CUBLAS_CHECK(cublasSgemm, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
  }
  void gemm(char op_a, char op_b, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *b,
            int ldb, std::complex<float> beta, std::complex<float> *c, int ldc) {
    CUBLAS_CHECK(cublasCgemm, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, cuscalar(alpha), cucplx(a), lda, cucplx(b), ldb, cuscalar(beta),
                 cucplx(c), ldc);
  }
  void gemm(char op_a, char op_b, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta, double *c,
            int ldc) {
    CUBLAS_CHECK(cublasDgemm, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
  }
  void gemm(char op_a, char op_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda,
            const std::complex<double> *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc) {
    CUBLAS_CHECK(cublasZgemm, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, cuscalar(alpha), cucplx(a), lda, cucplx(b), ldb, cuscalar(beta),
                 cucplx(c), ldc);
  }

  // gemm_batch
  void gemm_batch(char op_a, char op_b, int m, int n, int k, float alpha, const float **a, int lda, const float **b, int ldb, float beta, float **c,
                  int ldc, int batch_count) {
    CUBLAS_CHECK(cublasSgemmBatched, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc, batch_count);
  }
  void gemm_batch(char op_a, char op_b, int m, int n, int k, std::complex<float> alpha, const std::complex<float> **a, int lda,
                  const std::complex<float> **b, int ldb, std::complex<float> beta, std::complex<float> **c, int ldc, int batch_count) {
    CUBLAS_CHECK(cublasCgemmBatched, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, cuscalar(alpha), cucplx(a), lda, cucplx(b), ldb,
                 cuscalar(beta), cucplx(c), ldc, batch_count);
  }
  void gemm_batch(char op_a, char op_b, int m, int n, int k, double alpha, const double **a, int lda, const double **b, int ldb, double beta,
                  double **c, int ldc, int batch_count) {
    CUBLAS_CHECK(cublasDgemmBatched, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc, batch_count);
  }
  void gemm_batch(char op_a, char op_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double> **a, int lda,
                  const std::complex<double> **b, int ldb, std::complex<double> beta, std::complex<double> **c, int ldc, int batch_count) {
    CUBLAS_CHECK(cublasZgemmBatched, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, cuscalar(alpha), cucplx(a), lda, cucplx(b), ldb,
                 cuscalar(beta), cucplx(c), ldc, batch_count);
  }

  // gemm_vbatch
  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, float alpha, const float **a, int *lda, const float **b, int *ldb, float beta,
                   float **c, int *ldc, int batch_count) {
    cuda_gemm_vbatch(op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count);
  }
  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, std::complex<float> alpha, const std::complex<float> **a, int *lda,
                   const std::complex<float> **b, int *ldb, std::complex<float> beta, std::complex<float> **c, int *ldc, int batch_count) {
    magma_gemm_vbatch(op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count);
  }
  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, double alpha, const double **a, int *lda, const double **b, int *ldb, double beta,
                   double **c, int *ldc, int batch_count) {
    cuda_gemm_vbatch(op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count);
  }
  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, std::complex<double> alpha, const std::complex<double> **a, int *lda,
                   const std::complex<double> **b, int *ldb, std::complex<double> beta, std::complex<double> **c, int *ldc, int batch_count) {
    magma_gemm_vbatch(op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count);
  }

  // gemm_batch_strided
  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, float alpha, const float *a, int lda, int stride_a, const float *b, int ldb,
                          int stride_b, float beta, float *c, int ldc, int stride_c, int batch_count) {
    CUBLAS_CHECK(cublasSgemmStridedBatched, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, &alpha, a, lda, stride_a, b, ldb, stride_b, &beta, c,
                 ldc, stride_c, batch_count);
  }
  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda, int stride_a,
                          const std::complex<float> *b, int ldb, int stride_b, std::complex<float> beta, std::complex<float> *c, int ldc,
                          int stride_c, int batch_count) {
    CUBLAS_CHECK(cublasCgemmStridedBatched, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, cuscalar(alpha), cucplx(a), lda, stride_a, cucplx(b),
                 ldb, stride_b, cuscalar(beta), cucplx(c), ldc, stride_c, batch_count);
  }
  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, double alpha, const double *a, int lda, int stride_a, const double *b, int ldb,
                          int stride_b, double beta, double *c, int ldc, int stride_c, int batch_count) {
    CUBLAS_CHECK(cublasDgemmStridedBatched, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, &alpha, a, lda, stride_a, b, ldb, stride_b, &beta, c,
                 ldc, stride_c, batch_count);
  }
  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda, int stride_a,
                          const std::complex<double> *b, int ldb, int stride_b, std::complex<double> beta, std::complex<double> *c, int ldc,
                          int stride_c, int batch_count) {
    CUBLAS_CHECK(cublasZgemmStridedBatched, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, cuscalar(alpha), cucplx(a), lda, stride_a, cucplx(b),
                 ldb, stride_b, cuscalar(beta), cucplx(c), ldc, stride_c, batch_count);
  }

  // gemv
  void gemv(char op, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y, int incy) {
    CUBLAS_CHECK(cublasSgemv, get_cublas_op(op), m, n, &alpha, a, lda, x, incx, &beta, y, incy);
  }
  void gemv(char op, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *x, int incx,
            std::complex<float> beta, std::complex<float> *y, int incy) {
    CUBLAS_CHECK(cublasCgemv, get_cublas_op(op), m, n, cuscalar(alpha), cucplx(a), lda, cucplx(x), incx, cuscalar(beta), cucplx(y), incy);
  }
  void gemv(char op, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy) {
    CUBLAS_CHECK(cublasDgemv, get_cublas_op(op), m, n, &alpha, a, lda, x, incx, &beta, y, incy);
  }
  void gemv(char op, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda, const std::complex<double> *x, int incx,
            std::complex<double> beta, std::complex<double> *y, int incy) {
    CUBLAS_CHECK(cublasZgemv, get_cublas_op(op), m, n, cuscalar(alpha), cucplx(a), lda, cucplx(x), incx, cuscalar(beta), cucplx(y), incy);
  }

  // ger and gerc
  void ger(int m, int n, float alpha, const float *x, int incx, const float *y, int incy, float *a, int lda) {
    CUBLAS_CHECK(cublasSger, m, n, &alpha, x, incx, y, incy, a, lda);
  }
  void ger(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy,
           std::complex<float> *a, int lda) {
    CUBLAS_CHECK(cublasCgeru, m, n, cuscalar(alpha), cucplx(x), incx, cucplx(y), incy, cucplx(a), lda);
  }
  void gerc(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy,
            std::complex<float> *a, int lda) {
    CUBLAS_CHECK(cublasCgerc, m, n, cuscalar(alpha), cucplx(x), incx, cucplx(y), incy, cucplx(a), lda);
  }
  void ger(int m, int n, double alpha, const double *x, int incx, const double *y, int incy, double *a, int lda) {
    CUBLAS_CHECK(cublasDger, m, n, &alpha, x, incx, y, incy, a, lda);
  }
  void ger(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy,
           std::complex<double> *a, int lda) {
    CUBLAS_CHECK(cublasZgeru, m, n, cuscalar(alpha), cucplx(x), incx, cucplx(y), incy, cucplx(a), lda);
  }
  void gerc(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy,
            std::complex<double> *a, int lda) {
    CUBLAS_CHECK(cublasZgerc, m, n, cuscalar(alpha), cucplx(x), incx, cucplx(y), incy, cucplx(a), lda);
  }

  // scal
  void scal(int m, float alpha, float *x, int incx) { CUBLAS_CHECK(cublasSscal, m, &alpha, x, incx); }
  void scal(int m, std::complex<float> alpha, std::complex<float> *x, int incx) { CUBLAS_CHECK(cublasCscal, m, cuscalar(alpha), cucplx(x), incx); }
  void scal(int m, double alpha, double *x, int incx) { CUBLAS_CHECK(cublasDscal, m, &alpha, x, incx); }
  void scal(int m, std::complex<double> alpha, std::complex<double> *x, int incx) { CUBLAS_CHECK(cublasZscal, m, cuscalar(alpha), cucplx(x), incx); }

  // swap
  void swap(int n, float *x, int incx, float *y, int incy) { CUBLAS_CHECK(cublasSswap, n, x, incx, y, incy); } // NOLINT (this is a BLAS swap)
  void swap(int n, std::complex<float> *x, int incx, std::complex<float> *y, int incy) {                       // NOLINT (this is a BLAS swap)
    CUBLAS_CHECK(cublasCswap, n, cucplx(x), incx, cucplx(y), incy);
  }
  void swap(int n, double *x, int incx, double *y, int incy) { CUBLAS_CHECK(cublasDswap, n, x, incx, y, incy); } // NOLINT (this is a BLAS swap)
  void swap(int n, std::complex<double> *x, int incx, std::complex<double> *y, int incy) {                       // NOLINT (this is a BLAS swap)
    CUBLAS_CHECK(cublasZswap, n, cucplx(x), incx, cucplx(y), incy);
  }

  // getrf_batch
  void getrf_batch(int n, float **a_array, int lda, int *ipiv_array, int *info_array, int batch_size) {
    CUBLAS_CHECK(cublasSgetrfBatched, n, a_array, lda, ipiv_array, info_array, batch_size);
  }
  void getrf_batch(int n, std::complex<float> **a_array, int lda, int *ipiv_array, int *info_array, int batch_size) {
    CUBLAS_CHECK(cublasCgetrfBatched, n, cucplx(a_array), lda, ipiv_array, info_array, batch_size);
  }
  void getrf_batch(int n, double **a_array, int lda, int *ipiv_array, int *info_array, int batch_size) {
    CUBLAS_CHECK(cublasDgetrfBatched, n, a_array, lda, ipiv_array, info_array, batch_size);
  }
  void getrf_batch(int n, std::complex<double> **a_array, int lda, int *ipiv_array, int *info_array, int batch_size) {
    CUBLAS_CHECK(cublasZgetrfBatched, n, cucplx(a_array), lda, ipiv_array, info_array, batch_size);
  }

  // getri_batch
  void getri_batch(int n, float **a_array, int lda, int const *ipiv_array, float **c_array, int ldc, int *info_array, int batch_size) {
    CUBLAS_CHECK(cublasSgetriBatched, n, a_array, lda, ipiv_array, c_array, ldc, info_array, batch_size);
  }
  void getri_batch(int n, std::complex<float> **a_array, int lda, int const *ipiv_array, std::complex<float> **c_array, int ldc, int *info_array,
                   int batch_size) {
    CUBLAS_CHECK(cublasCgetriBatched, n, cucplx(a_array), lda, ipiv_array, cucplx(c_array), ldc, info_array, batch_size);
  }
  void getri_batch(int n, double **a_array, int lda, int const *ipiv_array, double **c_array, int ldc, int *info_array, int batch_size) {
    CUBLAS_CHECK(cublasDgetriBatched, n, a_array, lda, ipiv_array, c_array, ldc, info_array, batch_size);
  }
  void getri_batch(int n, std::complex<double> **a_array, int lda, int const *ipiv_array, std::complex<double> **c_array, int ldc, int *info_array,
                   int batch_size) {
    CUBLAS_CHECK(cublasZgetriBatched, n, cucplx(a_array), lda, ipiv_array, cucplx(c_array), ldc, info_array, batch_size);
  }

  // getrs_batch
  void getrs_batch(char op, int n, int nrhs, const float **a_array, int lda, int const *ipiv_array, float **b_array, int ldb, int &info,
                   int batch_size) {
    CUBLAS_CHECK(cublasSgetrsBatched, get_cublas_op(op), n, nrhs, a_array, lda, ipiv_array, b_array, ldb, &info, batch_size);
  }
  void getrs_batch(char op, int n, int nrhs, const std::complex<float> **a_array, int lda, int const *ipiv_array, std::complex<float> **b_array,
                   int ldb, int &info, int batch_size) {
    CUBLAS_CHECK(cublasCgetrsBatched, get_cublas_op(op), n, nrhs, cucplx(a_array), lda, ipiv_array, cucplx(b_array), ldb, &info, batch_size);
  }
  void getrs_batch(char op, int n, int nrhs, const double **a_array, int lda, int const *ipiv_array, double **b_array, int ldb, int &info,
                   int batch_size) {
    CUBLAS_CHECK(cublasDgetrsBatched, get_cublas_op(op), n, nrhs, a_array, lda, ipiv_array, b_array, ldb, &info, batch_size);
  }
  void getrs_batch(char op, int n, int nrhs, const std::complex<double> **a_array, int lda, int const *ipiv_array, std::complex<double> **b_array,
                   int ldb, int &info, int batch_size) {
    CUBLAS_CHECK(cublasZgetrsBatched, get_cublas_op(op), n, nrhs, cucplx(a_array), lda, ipiv_array, cucplx(b_array), ldb, &info, batch_size);
  }

  // geqrf_batch
  void geqrf_batch(int n, int m, float **a_array, int lda, float **tau_array, int &info, int batch_size) {
    CUBLAS_CHECK(cublasSgeqrfBatched, n, m, a_array, lda, tau_array, &info, batch_size);
  }
  void geqrf_batch(int n, int m, std::complex<float> **a_array, int lda, std::complex<float> **tau_array, int &info, int batch_size) {
    CUBLAS_CHECK(cublasCgeqrfBatched, n, m, cucplx(a_array), lda, cucplx(tau_array), &info, batch_size);
  }
  void geqrf_batch(int n, int m, double **a_array, int lda, double **tau_array, int &info, int batch_size) {
    CUBLAS_CHECK(cublasDgeqrfBatched, n, m, a_array, lda, tau_array, &info, batch_size);
  }
  void geqrf_batch(int n, int m, std::complex<double> **a_array, int lda, std::complex<double> **tau_array, int &info, int batch_size) {
    CUBLAS_CHECK(cublasZgeqrfBatched, n, m, cucplx(a_array), lda, cucplx(tau_array), &info, batch_size);
  }

} // namespace nda::blas::device
