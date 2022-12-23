// Copyright (c) 2019-2021 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Olivier Parcollet, Nils Wentzell

#include <nda/nda.hpp>
#include <nda/exceptions.hpp>
#include "cxx_interface.hpp"

// Extracted from Reference Lapack (https://github.com/Reference-LAPACK):
#include "cuda_runtime.h"
#include "cublas_v2.h"

#ifdef NDA_HAVE_MAGMA
#include "magma_v2.h"
#endif

#include <string>
#include <vector>

using namespace std::string_literals;

namespace nda::blas::device {

  constexpr cublasOperation_t get_cublas_op(char op) {
    switch (op) {
      case 'N': return CUBLAS_OP_N; break;
      case 'T': return CUBLAS_OP_T; break;
      case 'C': return CUBLAS_OP_C; break;
      default: std::terminate(); return {};
    }
  }

  // Get CuBlas Handle, Used by all routines
  auto &get_handle() {
    struct handle_t {
      handle_t() { cublasCreate(&h); }
      ~handle_t() { cublasDestroy(h); }
      operator cublasHandle_t() { return h; }

      private:
      cublasHandle_t h = {};
    };
    static handle_t h = {};
    return h;
  }

#ifdef NDA_HAVE_MAGMA
  constexpr magma_trans_t get_magma_op(char op) {
    switch (op) {
      case 'N': return MagmaNoTrans; break;
      case 'T': return MagmaTrans; break;
      case 'C': return MagmaConjTrans; break;
      default: std::terminate(); return {};
    }
  }

  // Get Magma queue, Used by all magma routines
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

  /// Global option to turn on/off the cudaDeviceSynchronize after cublas library calls
  static bool synchronize = true;
#define CUBLAS_CHECK(X, ...)                                                                                                                         \
  auto err = X(get_handle(), __VA_ARGS__);                                                                                                           \
  if (synchronize) cudaDeviceSynchronize();                                                                                                          \
  if (err != CUBLAS_STATUS_SUCCESS) NDA_RUNTIME_ERROR << AS_STRING(X) + " failed with error code "s + std::to_string(err);

  inline auto *cucplx(std::complex<double> *c) { return reinterpret_cast<cuDoubleComplex *>(c); }                // NOLINT
  inline auto *cucplx(std::complex<double> const *c) { return reinterpret_cast<const cuDoubleComplex *>(c); }    // NOLINT
  inline auto **cucplx(std::complex<double> **c) { return reinterpret_cast<cuDoubleComplex **>(c); }             // NOLINT
  inline auto **cucplx(std::complex<double> const **c) { return reinterpret_cast<const cuDoubleComplex **>(c); } // NOLINT

  void gemm(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, const double *B, int LDB, double beta, double *C,
            int LDC) {
    CUBLAS_CHECK(cublasDgemm, get_cublas_op(op_a), get_cublas_op(op_b), M, N, K, &alpha, A, LDA, B, LDB, &beta, C, LDC);
  }
  void gemm(char op_a, char op_b, int M, int N, int K, std::complex<double> alpha, const std::complex<double> *A, int LDA,
            const std::complex<double> *B, int LDB, std::complex<double> beta, std::complex<double> *C, int LDC) {
    auto alpha_cu = cuDoubleComplex{alpha.real(), alpha.imag()};
    auto beta_cu  = cuDoubleComplex{beta.real(), beta.imag()};
    CUBLAS_CHECK(cublasZgemm, get_cublas_op(op_a), get_cublas_op(op_b), M, N, K, &alpha_cu, cucplx(A), LDA, cucplx(B), LDB, &beta_cu, cucplx(C), LDC);
  }

  void gemm_batch(char op_a, char op_b, int M, int N, int K, double alpha, const double **A, int LDA, const double **B, int LDB, double beta,
                  double **C, int LDC, int batch_count) {
    CUBLAS_CHECK(cublasDgemmBatched, get_cublas_op(op_a), get_cublas_op(op_b), M, N, K, &alpha, A, LDA, B, LDB, &beta, C, LDC, batch_count);
  }
  void gemm_batch(char op_a, char op_b, int M, int N, int K, std::complex<double> alpha, const std::complex<double> **A, int LDA,
                  const std::complex<double> **B, int LDB, std::complex<double> beta, std::complex<double> **C, int LDC, int batch_count) {
    auto alpha_cu = cuDoubleComplex{alpha.real(), alpha.imag()};
    auto beta_cu  = cuDoubleComplex{beta.real(), beta.imag()};
    CUBLAS_CHECK(cublasZgemmBatched, get_cublas_op(op_a), get_cublas_op(op_b), M, N, K, &alpha_cu, cucplx(A), LDA, cucplx(B), LDB, &beta_cu,
                 cucplx(C), LDC, batch_count);
  }

  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, double alpha, const double **A, int *LDA, const double **B, int *LDB, double beta,
                   double **C, int *LDC, int batch_count) {
#ifdef NDA_HAVE_MAGMA
    magmablas_dgemm_vbatched(get_magma_op(op_a), get_magma_op(op_b), M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC, batch_count, get_magma_queue());
    if (synchronize) magma_queue_sync(get_magma_queue());
    if (synchronize) cudaDeviceSynchronize();
#else
    NDA_RUNTIME_ERROR << "nda::blas::cuda::gemmv_batch requires Magma [https://icl.cs.utk.edu/magma/]. Configure nda with -DUse_Magma=ON";
#endif
  }
  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, std::complex<double> alpha, const std::complex<double> **A, int *LDA,
                   const std::complex<double> **B, int *LDB, std::complex<double> beta, std::complex<double> **C, int *LDC, int batch_count) {
#ifdef NDA_HAVE_MAGMA
    auto alpha_cu = cuDoubleComplex{alpha.real(), alpha.imag()};
    auto beta_cu  = cuDoubleComplex{beta.real(), beta.imag()};
    magmablas_zgemm_vbatched(get_magma_op(op_a), get_magma_op(op_b), M, N, K, alpha_cu, cucplx(A), LDA, cucplx(B), LDB, beta_cu, cucplx(C), LDC,
                             batch_count, get_magma_queue());
    if (synchronize) magma_queue_sync(get_magma_queue());
    if (synchronize) cudaDeviceSynchronize();
#else
    NDA_RUNTIME_ERROR << "nda::blas::cuda::gemmv_batch requires Magma [https://icl.cs.utk.edu/magma/]. Configure nda with -DUse_Magma=ON";
#endif
  }

  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, int strideA, const double *B, int LDB,
                          int strideB, double beta, double *C, int LDC, int strideC, int batch_count) {
    CUBLAS_CHECK(cublasDgemmStridedBatched, get_cublas_op(op_a), get_cublas_op(op_b), M, N, K, &alpha, A, LDA, strideA, B, LDB, strideB, &beta, C,
                 LDC, strideC, batch_count);
  }
  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, std::complex<double> alpha, const std::complex<double> *A, int LDA, int strideA,
                          const std::complex<double> *B, int LDB, int strideB, std::complex<double> beta, std::complex<double> *C, int LDC,
                          int strideC, int batch_count) {
    auto alpha_cu = cuDoubleComplex{alpha.real(), alpha.imag()};
    auto beta_cu  = cuDoubleComplex{beta.real(), beta.imag()};
    CUBLAS_CHECK(cublasZgemmStridedBatched, get_cublas_op(op_a), get_cublas_op(op_b), M, N, K, &alpha_cu, cucplx(A), LDA, strideA, cucplx(B), LDB,
                 strideB, &beta_cu, cucplx(C), LDC, strideC, batch_count);
  }

  void axpy(int N, double alpha, const double *x, int incx, double *Y, int incy) { cublasDaxpy(get_handle(), N, &alpha, x, incx, Y, incy); }
  void axpy(int N, std::complex<double> alpha, const std::complex<double> *x, int incx, std::complex<double> *Y, int incy) {
    CUBLAS_CHECK(cublasZaxpy, N, cucplx(&alpha), cucplx(x), incx, cucplx(Y), incy);
  }

  void copy(int N, const double *x, int incx, double *Y, int incy) { cublasDcopy(get_handle(), N, x, incx, Y, incy); }
  void copy(int N, const std::complex<double> *x, int incx, std::complex<double> *Y, int incy) {
    CUBLAS_CHECK(cublasZcopy, N, cucplx(x), incx, cucplx(Y), incy);
  }

  double dot(int M, const double *x, int incx, const double *Y, int incy) {
    double res{};
    CUBLAS_CHECK(cublasDdot, M, x, incx, Y, incy, &res);
    return res;
  }
  std::complex<double> dot(int M, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy) {
    cuDoubleComplex res;
    CUBLAS_CHECK(cublasZdotu, M, cucplx(x), incx, cucplx(Y), incy, &res);
    return {res.x, res.y};
  }
  std::complex<double> dotc(int M, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy) {
    cuDoubleComplex res;
    CUBLAS_CHECK(cublasZdotc, M, cucplx(x), incx, cucplx(Y), incy, &res);
    return {res.x, res.y};
  }

  void gemv(char op, int M, int N, double alpha, const double *A, int LDA, const double *x, int incx, double beta, double *Y, int incy) {
    CUBLAS_CHECK(cublasDgemv, get_cublas_op(op), M, N, &alpha, A, LDA, x, incx, &beta, Y, incy);
  }
  void gemv(char op, int M, int N, std::complex<double> alpha, const std::complex<double> *A, int LDA, const std::complex<double> *x, int incx,
            std::complex<double> beta, std::complex<double> *Y, int incy) {
    CUBLAS_CHECK(cublasZgemv, get_cublas_op(op), M, N, cucplx(&alpha), cucplx(A), LDA, cucplx(x), incx, cucplx(&beta), cucplx(Y), incy);
  }

  void ger(int M, int N, double alpha, const double *x, int incx, const double *Y, int incy, double *A, int LDA) {
    CUBLAS_CHECK(cublasDger, M, N, &alpha, x, incx, Y, incy, A, LDA);
  }
  void ger(int M, int N, std::complex<double> alpha, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy,
           std::complex<double> *A, int LDA) {
    CUBLAS_CHECK(cublasZgeru, M, N, cucplx(&alpha), cucplx(x), incx, cucplx(Y), incy, cucplx(A), LDA);
  }

  void scal(int M, double alpha, double *x, int incx) { CUBLAS_CHECK(cublasDscal, M, &alpha, x, incx); }
  void scal(int M, std::complex<double> alpha, std::complex<double> *x, int incx) { CUBLAS_CHECK(cublasZscal, M, cucplx(&alpha), cucplx(x), incx); }

  void swap(int N, double *x, int incx, double *Y, int incy) { CUBLAS_CHECK(cublasDswap, N, x, incx, Y, incy); }
  void swap(int N, std::complex<double> *x, int incx, std::complex<double> *Y, int incy) {
    CUBLAS_CHECK(cublasZswap, N, cucplx(x), incx, cucplx(Y), incy);
  }

} // namespace nda::blas::device
