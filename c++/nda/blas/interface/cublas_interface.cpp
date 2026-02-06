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

  // Global option to turn on/off the cudaDeviceSynchronize after cublas library calls.
  static bool synchronize = true; // NOLINT  (global option is on purpose)

// Macro to check cublas calls.
#define CUBLAS_CHECK(X, ...)                                                                                                                         \
  {                                                                                                                                                  \
    auto err = X(get_handle(), __VA_ARGS__);                                                                                                         \
    if (err != CUBLAS_STATUS_SUCCESS) {                                                                                                              \
      NDA_RUNTIME_ERROR << AS_STRING(X) << " failed \n"                                                                                              \
                        << " cublasGetStatusName: " << cublasGetStatusName(err) << "\n"                                                              \
                        << " cublasGetStatusString: " << cublasGetStatusString(err) << "\n";                                                         \
    }                                                                                                                                                \
    if (synchronize) {                                                                                                                               \
      auto errsync = cudaDeviceSynchronize();                                                                                                        \
      if (errsync != cudaSuccess) {                                                                                                                  \
        NDA_RUNTIME_ERROR << " cudaDeviceSynchronize failed after call to: " << AS_STRING(X) << "\n"                                                 \
                          << " cudaGetErrorName: " << cudaGetErrorName(errsync) << "\n"                                                              \
                          << " cudaGetErrorString: " << cudaGetErrorString(errsync) << "\n";                                                         \
      }                                                                                                                                              \
    }                                                                                                                                                \
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
      if (synchronize) cudaDeviceSynchronize();
    }
#else
    template <typename T>
    void magma_gemm_vbatch(char, char, int *, int *, int *, T, const T **, int *, const T **, int *, T, T **, int *, int) {
      NDA_RUNTIME_ERROR << "nda::blas::device::gemmv_batch with complex types requires Magma. Configure nda with -DMagmaSupport=ON";
    }
#endif

  } // namespace

  void axpy(int n, double alpha, const double *x, int incx, double *y, int incy) { cublasDaxpy(get_handle(), n, &alpha, x, incx, y, incy); }
  void axpy(int n, std::complex<double> alpha, const std::complex<double> *x, int incx, std::complex<double> *y, int incy) {
    auto alpha_cu = cucplx(alpha);
    CUBLAS_CHECK(cublasZaxpy, n, &alpha_cu, cucplx(x), incx, cucplx(y), incy);
  }

  void copy(int n, const double *x, int incx, double *y, int incy) { cublasDcopy(get_handle(), n, x, incx, y, incy); }
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
    CUBLAS_CHECK(cublasCgemm, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, cucplx(&alpha), cucplx(a), lda, cucplx(b), ldb, cucplx(&beta),
                 cucplx(c), ldc);
  }
  void gemm(char op_a, char op_b, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta, double *c,
            int ldc) {
    CUBLAS_CHECK(cublasDgemm, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
  }
  void gemm(char op_a, char op_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda,
            const std::complex<double> *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc) {
    CUBLAS_CHECK(cublasZgemm, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, cucplx(&alpha), cucplx(a), lda, cucplx(b), ldb, cucplx(&beta),
                 cucplx(c), ldc);
  }

  // gemm_batch
  void gemm_batch(char op_a, char op_b, int m, int n, int k, float alpha, const float **a, int lda, const float **b, int ldb, float beta, float **c,
                  int ldc, int batch_count) {
    CUBLAS_CHECK(cublasSgemmBatched, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc, batch_count);
  }
  void gemm_batch(char op_a, char op_b, int m, int n, int k, std::complex<float> alpha, const std::complex<float> **a, int lda,
                  const std::complex<float> **b, int ldb, std::complex<float> beta, std::complex<float> **c, int ldc, int batch_count) {
    CUBLAS_CHECK(cublasCgemmBatched, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, cucplx(&alpha), cucplx(a), lda, cucplx(b), ldb, cucplx(&beta),
                 cucplx(c), ldc, batch_count);
  }
  void gemm_batch(char op_a, char op_b, int m, int n, int k, double alpha, const double **a, int lda, const double **b, int ldb, double beta,
                  double **c, int ldc, int batch_count) {
    CUBLAS_CHECK(cublasDgemmBatched, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc, batch_count);
  }
  void gemm_batch(char op_a, char op_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double> **a, int lda,
                  const std::complex<double> **b, int ldb, std::complex<double> beta, std::complex<double> **c, int ldc, int batch_count) {
    CUBLAS_CHECK(cublasZgemmBatched, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, cucplx(&alpha), cucplx(a), lda, cucplx(b), ldb, cucplx(&beta),
                 cucplx(c), ldc, batch_count);
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
    CUBLAS_CHECK(cublasCgemmStridedBatched, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, cucplx(&alpha), cucplx(a), lda, stride_a, cucplx(b),
                 ldb, stride_b, cucplx(&beta), cucplx(c), ldc, stride_c, batch_count);
  }
  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, double alpha, const double *a, int lda, int stride_a, const double *b, int ldb,
                          int stride_b, double beta, double *c, int ldc, int stride_c, int batch_count) {
    CUBLAS_CHECK(cublasDgemmStridedBatched, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, &alpha, a, lda, stride_a, b, ldb, stride_b, &beta, c,
                 ldc, stride_c, batch_count);
  }
  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda, int stride_a,
                          const std::complex<double> *b, int ldb, int stride_b, std::complex<double> beta, std::complex<double> *c, int ldc,
                          int stride_c, int batch_count) {
    CUBLAS_CHECK(cublasZgemmStridedBatched, get_cublas_op(op_a), get_cublas_op(op_b), m, n, k, cucplx(&alpha), cucplx(a), lda, stride_a, cucplx(b),
                 ldb, stride_b, cucplx(&beta), cucplx(c), ldc, stride_c, batch_count);
  }

  // gemv
  void gemv(char op, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y, int incy) {
    CUBLAS_CHECK(cublasSgemv, get_cublas_op(op), m, n, &alpha, a, lda, x, incx, &beta, y, incy);
  }
  void gemv(char op, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *x, int incx,
            std::complex<float> beta, std::complex<float> *y, int incy) {
    CUBLAS_CHECK(cublasCgemv, get_cublas_op(op), m, n, cucplx(&alpha), cucplx(a), lda, cucplx(x), incx, cucplx(&beta), cucplx(y), incy);
  }
  void gemv(char op, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy) {
    CUBLAS_CHECK(cublasDgemv, get_cublas_op(op), m, n, &alpha, a, lda, x, incx, &beta, y, incy);
  }
  void gemv(char op, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda, const std::complex<double> *x, int incx,
            std::complex<double> beta, std::complex<double> *y, int incy) {
    CUBLAS_CHECK(cublasZgemv, get_cublas_op(op), m, n, cucplx(&alpha), cucplx(a), lda, cucplx(x), incx, cucplx(&beta), cucplx(y), incy);
  }

  // ger and gerc
  void ger(int m, int n, float alpha, const float *x, int incx, const float *y, int incy, float *a, int lda) {
    CUBLAS_CHECK(cublasSger, m, n, &alpha, x, incx, y, incy, a, lda);
  }
  void ger(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy,
           std::complex<float> *a, int lda) {
    CUBLAS_CHECK(cublasCgeru, m, n, cucplx(&alpha), cucplx(x), incx, cucplx(y), incy, cucplx(a), lda);
  }
  void gerc(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy,
            std::complex<float> *a, int lda) {
    CUBLAS_CHECK(cublasCgerc, m, n, cucplx(&alpha), cucplx(x), incx, cucplx(y), incy, cucplx(a), lda);
  }
  void ger(int m, int n, double alpha, const double *x, int incx, const double *y, int incy, double *a, int lda) {
    CUBLAS_CHECK(cublasDger, m, n, &alpha, x, incx, y, incy, a, lda);
  }
  void ger(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy,
           std::complex<double> *a, int lda) {
    CUBLAS_CHECK(cublasZgeru, m, n, cucplx(&alpha), cucplx(x), incx, cucplx(y), incy, cucplx(a), lda);
  }
  void gerc(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy,
            std::complex<double> *a, int lda) {
    CUBLAS_CHECK(cublasZgerc, m, n, cucplx(&alpha), cucplx(x), incx, cucplx(y), incy, cucplx(a), lda);
  }

  // scal
  void scal(int m, float alpha, float *x, int incx) { CUBLAS_CHECK(cublasSscal, m, &alpha, x, incx); }
  void scal(int m, std::complex<float> alpha, std::complex<float> *x, int incx) { CUBLAS_CHECK(cublasCscal, m, cucplx(&alpha), cucplx(x), incx); }
  void scal(int m, double alpha, double *x, int incx) { CUBLAS_CHECK(cublasDscal, m, &alpha, x, incx); }
  void scal(int m, std::complex<double> alpha, std::complex<double> *x, int incx) { CUBLAS_CHECK(cublasZscal, m, cucplx(&alpha), cucplx(x), incx); }

  void swap(int n, double *x, int incx, double *y, int incy) { CUBLAS_CHECK(cublasDswap, n, x, incx, y, incy); } // NOLINT (this is a BLAS swap)
  void swap(int n, std::complex<double> *x, int incx, std::complex<double> *y, int incy) {                       // NOLINT (this is a BLAS swap)
    CUBLAS_CHECK(cublasZswap, n, cucplx(x), incx, cucplx(y), incy);
  }

} // namespace nda::blas::device
