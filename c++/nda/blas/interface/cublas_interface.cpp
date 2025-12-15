// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Implementation details for blas/interface/cublas_interface.hpp.
 */

#include <mpi/mpi.hpp>
#include <nda/nda.hpp>
#include <nda/device.hpp>
#include "./cublas_interface.hpp"
#include "../tools.hpp"
#include "../../device.hpp"
#include "../../exceptions.hpp"

#ifdef NDA_HAVE_MAGMA
#include "magma_v2.h"
#endif

namespace nda::blas::device {

  static const char *_cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
      case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
      case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
      case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
      case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
      case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
      case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
      case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
      case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
      case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
      case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
  }

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
      std::cerr << AS_STRING(X) << " failed with error code: " << std::to_string(err) << ", error message: " << _cudaGetErrorEnum(err) << std::endl; \
      mpi::communicator{}.abort(11);                                                                                                                 \
    }                                                                                                                                                \
    if (synchronize) {                                                                                                                               \
      auto errsync = cudaDeviceSynchronize();                                                                                                        \
      if (errsync != cudaSuccess) {                                                                                                                  \
        std::cerr << " cudaDeviceSynchronize failed after call to: " << AS_STRING(X) " \n "                                                          \
                  << " cudaGetErrorName: " << std::string(cudaGetErrorName(errsync)) << "\n"                                                         \
                  << " cudaGetErrorString: " << std::string(cudaGetErrorString(errsync)) << std::endl;                                               \
        mpi::communicator{}.abort(11);                                                                                                               \
      }                                                                                                                                              \
    }                                                                                                                                                \
  }

#define _gemm_(FUN, TYPE)                                                                                                                            \
  void gemm(char op_a, char op_b, int M, int N, int K, TYPE alpha, const TYPE *A, int LDA, const TYPE *B, int LDB, TYPE beta, TYPE *C, int LDC) {    \
    auto alpha_cu = to_cublas(alpha);                                                                                                                \
    auto beta_cu  = to_cublas(beta);                                                                                                                 \
    CUBLAS_CHECK(FUN, get_cublas_op(op_a), get_cublas_op(op_b), M, N, K, &alpha_cu, to_cublas(A), LDA, to_cublas(B), LDB, &beta_cu, to_cublas(C),    \
                 LDC);                                                                                                                               \
  }
  _gemm_(cublasSgemm, float) _gemm_(cublasDgemm, double) _gemm_(cublasCgemm, fcomplex) _gemm_(cublasZgemm, dcomplex)

#define _gemm_batch_(FUN, TYPE)                                                                                                                      \
  void gemm_batch(char op_a, char op_b, int M, int N, int K, TYPE alpha, const TYPE **A, int LDA, const TYPE **B, int LDB, TYPE beta, TYPE **C,      \
                  int LDC, int batch_count) {                                                                                                        \
    auto alpha_cu = to_cublas(alpha);                                                                                                                \
    auto beta_cu  = to_cublas(beta);                                                                                                                 \
    CUBLAS_CHECK(FUN, get_cublas_op(op_a), get_cublas_op(op_b), M, N, K, &alpha_cu, to_cublas(A), LDA, to_cublas(B), LDB, &beta_cu, to_cublas(C),    \
                 LDC, batch_count);                                                                                                                  \
  }
     _gemm_batch_(cublasSgemmBatched, float) _gemm_batch_(cublasDgemmBatched, double) _gemm_batch_(cublasCgemmBatched, fcomplex)
        _gemm_batch_(cublasZgemmBatched, dcomplex)

#ifdef NDA_HAVE_MAGMA
#define _gemm_vbatch_(FUN, TYPE)                                                                                                                     \
  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, TYPE alpha, const TYPE **A, int *LDA, const TYPE **B, int *LDB, TYPE beta,          \
                   TYPE **C, int *LDC, int batch_count) {                                                                                            \
    auto alpha_cu = to_cublas(alpha);                                                                                                                \
    auto beta_cu  = to_cublas(beta);                                                                                                                 \
    magmablas_zgemm_vbatched(get_magma_op(op_a), get_magma_op(op_b), M, N, K, alpha_cu, to_cublas(A), LDA, to_cublas(B), LDB, beta_cu, to_cublas(C), \
                             LDC, batch_count, get_magma_queue());                                                                                   \
    if (synchronize) magma_queue_sync(get_magma_queue());                                                                                            \
    if (synchronize) cudaDeviceSynchronize();                                                                                                        \
  }
           _gemm_vbatch_(magmablas_sgemm_vbatched, float) _gemm_vbatch_(magmablas_dgemm_vbatched, double)
              _gemm_vbatch_(magmablas_cgemm_vbatched, fcomplex) _gemm_vbatch_(magmablas_zgemm_vbatched, dcomplex)
#endif

#define _gemm_batch_strided_(FUN, TYPE)                                                                                                              \
  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, TYPE alpha, const TYPE *A, int LDA, int strideA, const TYPE *B, int LDB,        \
                          int strideB, TYPE beta, TYPE *C, int LDC, int strideC, int batch_count) {                                                  \
    auto alpha_cu = to_cublas(alpha);                                                                                                                \
    auto beta_cu  = to_cublas(beta);                                                                                                                 \
    CUBLAS_CHECK(FUN, get_cublas_op(op_a), get_cublas_op(op_b), M, N, K, &alpha_cu, to_cublas(A), LDA, strideA, to_cublas(B), LDB, strideB,          \
                 &beta_cu, to_cublas(C), LDC, strideC, batch_count);                                                                                 \
  }
                 _gemm_batch_strided_(cublasSgemmStridedBatched, float) _gemm_batch_strided_(cublasDgemmStridedBatched, double)
                    _gemm_batch_strided_(cublasCgemmStridedBatched, fcomplex) _gemm_batch_strided_(cublasZgemmStridedBatched, dcomplex)

#define _axpy_(FUN, TYPE)                                                                                                                            \
  void axpy(int N, TYPE alpha, const TYPE *x, int incx, TYPE *Y, int incy) {                                                                         \
    CUBLAS_CHECK(FUN, N, to_cublas(&alpha), to_cublas(x), incx, to_cublas(Y), incy);                                                                 \
  }
                       _axpy_(cublasSaxpy, float) _axpy_(cublasDaxpy, double) _axpy_(cublasCaxpy, fcomplex) _axpy_(cublasZaxpy, dcomplex)

#define _copy_(FUN, TYPE)                                                                                                                            \
  void copy(int N, const TYPE *x, int incx, TYPE *Y, int incy) { CUBLAS_CHECK(FUN, N, to_cublas(x), incx, to_cublas(Y), incy); }
                          _copy_(cublasScopy, float) _copy_(cublasDcopy, double) _copy_(cublasCcopy, fcomplex) _copy_(cublasZcopy, dcomplex)

                             float dot(int M, const float *x, int incx, const float *Y, int incy) {
    float res{};
    CUBLAS_CHECK(cublasSdot, M, x, incx, Y, incy, &res);
    return res;
  }
  double dot(int M, const double *x, int incx, const double *Y, int incy) {
    double res{};
    CUBLAS_CHECK(cublasDdot, M, x, incx, Y, incy, &res);
    return res;
  }
  fcomplex dot(int M, const fcomplex *x, int incx, const fcomplex *Y, int incy) {
    cuComplex res;
    CUBLAS_CHECK(cublasCdotu, M, to_cublas(x), incx, to_cublas(Y), incy, &res);
    return {res.x, res.y};
  }
  dcomplex dot(int M, const dcomplex *x, int incx, const dcomplex *Y, int incy) {
    cuDoubleComplex res;
    CUBLAS_CHECK(cublasZdotu, M, to_cublas(x), incx, to_cublas(Y), incy, &res);
    return {res.x, res.y};
  }

  float dotc(int M, const float *x, int incx, const float *Y, int incy) {
    float res{};
    CUBLAS_CHECK(cublasSdot, M, x, incx, Y, incy, &res);
    return res;
  }
  double dotc(int M, const double *x, int incx, const double *Y, int incy) {
    double res{};
    CUBLAS_CHECK(cublasDdot, M, x, incx, Y, incy, &res);
    return res;
  }
  fcomplex dotc(int M, const fcomplex *x, int incx, const fcomplex *Y, int incy) {
    cuComplex res;
    CUBLAS_CHECK(cublasCdotc, M, to_cublas(x), incx, to_cublas(Y), incy, &res);
    return {res.x, res.y};
  }
  dcomplex dotc(int M, const dcomplex *x, int incx, const dcomplex *Y, int incy) {
    cuDoubleComplex res;
    CUBLAS_CHECK(cublasZdotc, M, to_cublas(x), incx, to_cublas(Y), incy, &res);
    return {res.x, res.y};
  }

#define _gemv_(FUN, TYPE)                                                                                                                            \
  void gemv(char op, int M, int N, TYPE alpha, const TYPE *A, int LDA, const TYPE *x, int incx, TYPE beta, TYPE *Y, int incy) {                      \
    CUBLAS_CHECK(FUN, get_cublas_op(op), M, N, to_cublas(&alpha), to_cublas(A), LDA, to_cublas(x), incx, to_cublas(&beta), to_cublas(Y), incy);      \
  }
  _gemv_(cublasSgemv, float) _gemv_(cublasDgemv, double) _gemv_(cublasCgemv, fcomplex) _gemv_(cublasZgemv, dcomplex)

#define _ger_(FUN, TYPE)                                                                                                                             \
  void ger(int M, int N, TYPE alpha, const TYPE *x, int incx, const TYPE *Y, int incy, TYPE *A, int LDA) {                                           \
    CUBLAS_CHECK(FUN, M, N, to_cublas(&alpha), to_cublas(x), incx, to_cublas(Y), incy, to_cublas(A), LDA);                                           \
  }
     _ger_(cublasSger, float) _ger_(cublasDger, double) _ger_(cublasCgeru, fcomplex) _ger_(cublasZgeru, dcomplex)

#define _gerc_(FUN, TYPE)                                                                                                                            \
  void gerc(int M, int N, TYPE alpha, const TYPE *x, int incx, const TYPE *Y, int incy, TYPE *A, int LDA) {                                          \
    CUBLAS_CHECK(FUN, M, N, to_cublas(&alpha), to_cublas(x), incx, to_cublas(Y), incy, to_cublas(A), LDA);                                           \
  }
        _gerc_(cublasSger, float) _gerc_(cublasDger, double) _gerc_(cublasCgerc, fcomplex) _gerc_(cublasZgerc, dcomplex)

#define _scal_(FUN, TYPE)                                                                                                                            \
  void scal(int M, TYPE alpha, TYPE *x, int incx) { CUBLAS_CHECK(FUN, M, to_cublas(&alpha), to_cublas(x), incx); }
           _scal_(cublasSscal, float) _scal_(cublasDscal, double) _scal_(cublasCscal, fcomplex) _scal_(cublasZscal, dcomplex)

#define _swap_(FUN, TYPE)                                                                                                                            \
  void swap(int N, TYPE *x, int incx, TYPE *Y, int incy) { CUBLAS_CHECK(FUN, N, to_cublas(x), incx, to_cublas(Y), incy); }
              _swap_(cublasSswap, float) _swap_(cublasDswap, double) _swap_(cublasCswap, fcomplex) _swap_(cublasZswap, dcomplex)

#define _getrf_batched_(FUN, TYPE)                                                                                                                   \
  void getrf_batched(int N, TYPE **A_array, int lda, int *ipiv_array, int *info_array, int batchSize) {                                              \
    CUBLAS_CHECK(FUN, N, to_cublas(A_array), lda, ipiv_array, info_array, batchSize);                                                                \
  }
                 _getrf_batched_(cublasSgetrfBatched, float);
  _getrf_batched_(cublasDgetrfBatched, double);
  _getrf_batched_(cublasCgetrfBatched, std::complex<float>);
  _getrf_batched_(cublasZgetrfBatched, std::complex<double>);

#define _getri_batched_(FUN, TYPE)                                                                                                                   \
  void getri_batched(int N, TYPE **A_array, int lda, int const *ipiv_array, TYPE **C_array, int ldc, int *info_array, int batchSize) {               \
    CUBLAS_CHECK(FUN, N, to_cublas(A_array), lda, ipiv_array, to_cublas(C_array), ldc, info_array, batchSize);                                       \
  }
  _getri_batched_(cublasSgetriBatched, float);
  _getri_batched_(cublasDgetriBatched, double);
  _getri_batched_(cublasCgetriBatched, std::complex<float>);
  _getri_batched_(cublasZgetriBatched, std::complex<double>);

#define _getrs_batched_(FUN, TYPE)                                                                                                                   \
  void getrs_batched(char op, int N, int NRHS, const TYPE **A_array, int lda, int const *ipiv_array, TYPE **B_array, int ldb, int *info_array,       \
                     int batchSize) {                                                                                                                \
    CUBLAS_CHECK(FUN, get_cublas_op(op), N, NRHS, to_cublas(A_array), lda, ipiv_array, to_cublas(B_array), ldb, info_array, batchSize);              \
  }
  _getrs_batched_(cublasSgetrsBatched, float);
  _getrs_batched_(cublasDgetrsBatched, double);
  _getrs_batched_(cublasCgetrsBatched, std::complex<float>);
  _getrs_batched_(cublasZgetrsBatched, std::complex<double>);

#define _geqrf_batched_(FUN, TYPE)                                                                                                                   \
  void geqrf_batched(int N, int M, TYPE **A_array, int lda, TYPE **tau_array, int *info_array, int batchSize) {                                      \
    CUBLAS_CHECK(FUN, N, M, to_cublas(A_array), lda, to_cublas(tau_array), info_array, batchSize);                                                   \
  }
  _geqrf_batched_(cublasSgeqrfBatched, float);
  _geqrf_batched_(cublasDgeqrfBatched, double);
  _geqrf_batched_(cublasCgeqrfBatched, std::complex<float>);
  _geqrf_batched_(cublasZgeqrfBatched, std::complex<double>);

} // namespace nda::blas::device
