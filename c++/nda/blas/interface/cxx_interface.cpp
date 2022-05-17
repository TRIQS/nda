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

#include <nda/macros.hpp>
#include <nda/exceptions.hpp>
#include "cxx_interface.hpp"

// Extracted from Reference Lapack (https://github.com/Reference-LAPACK):
#include "cblas_f77.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <string>
#include <vector>

using namespace std::string_literals;

// Manual Include since cblas uses Fortan _sub to wrap function again
#define F77_ddot F77_GLOBAL(ddot, DDOT)
#define F77_zdotu F77_GLOBAL(zdotu, DDOT)
#define F77_zdotc F77_GLOBAL(zdotc, DDOT)
extern "C" {
double F77_ddot(FINT, const double *, FINT, const double *, FINT);
std::complex<double> F77_zdotu(FINT, const double *, FINT, const double *, FINT); // NOLINT
std::complex<double> F77_zdotc(FINT, const double *, FINT, const double *, FINT); // NOLINT
#ifdef HAVE_GEMM_BATCH
void dgemm_batch(FCHAR, FCHAR, FINT, FINT, FINT, const double *, const double **, FINT, const double **, FINT, const double *, double **, FINT, FINT,
                 FINT);
void zgemm_batch(FCHAR, FCHAR, FINT, FINT, FINT, const double *, const double **, FINT, const double **, FINT, const double *, double **, FINT, FINT,
                 FINT);
void dgemm_batch_strided(FCHAR, FCHAR, FINT, FINT, FINT, const double *, const double *, FINT, FINT, const double *, FINT, FINT, const double *,
                         double *, FINT, FINT, FINT);
void zgemm_batch_strided(FCHAR, FCHAR, FINT, FINT, FINT, const double *, const double *, FINT, FINT, const double *, FINT, FINT, const double *,
                         double *, FINT, FINT, FINT);
#endif
}

namespace nda::blas::f77 {

  inline auto *blacplx(std::complex<double> *c) { return reinterpret_cast<double *>(c); }                // NOLINT
  inline auto *blacplx(std::complex<double> const *c) { return reinterpret_cast<const double *>(c); }    // NOLINT
  inline auto **blacplx(std::complex<double> **c) { return reinterpret_cast<double **>(c); }             // NOLINT
  inline auto **blacplx(std::complex<double> const **c) { return reinterpret_cast<const double **>(c); } // NOLINT

  void axpy(int N, double alpha, const double *x, int incx, double *Y, int incy) { F77_daxpy(&N, &alpha, x, &incx, Y, &incy); }
  void axpy(int N, std::complex<double> alpha, const std::complex<double> *x, int incx, std::complex<double> *Y, int incy) {
    F77_zaxpy(&N, blacplx(&alpha), blacplx(x), &incx, blacplx(Y), &incy);
  }
  // No Const In Wrapping!
  void copy(int N, const double *x, int incx, double *Y, int incy) { F77_dcopy(&N, x, &incx, Y, &incy); }
  void copy(int N, const std::complex<double> *x, int incx, std::complex<double> *Y, int incy) {
    F77_zcopy(&N, blacplx(x), &incx, blacplx(Y), &incy);
  }

  double dot(int M, const double *x, int incx, const double *Y, int incy) { return F77_ddot(&M, x, &incx, Y, &incy); }
  std::complex<double> dot(int M, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy) {
    return F77_zdotu(&M, blacplx(x), &incx, blacplx(Y), &incy);
  }
  std::complex<double> dotc(int M, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy) {
    return F77_zdotc(&M, blacplx(x), &incx, blacplx(Y), &incy);
  }

  void gemm(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, const double *B, int LDB, double beta, double *C,
            int LDC) {
    F77_dgemm(&op_a, &op_b, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, C, &LDC);
  }
  void gemm(char op_a, char op_b, int M, int N, int K, std::complex<double> alpha, const std::complex<double> *A, int LDA,
            const std::complex<double> *B, int LDB, std::complex<double> beta, std::complex<double> *C, int LDC) {
    F77_zgemm(&op_a, &op_b, &M, &N, &K, blacplx(&alpha), blacplx(A), &LDA, blacplx(B), &LDB, blacplx(&beta), blacplx(C), &LDC);
  }

  void gemm_batch(char op_a, char op_b, int M, int N, int K, double alpha, const double **A, int LDA, const double **B, int LDB, double beta,
                  double **C, int LDC, int batch_count) {
#ifdef HAVE_GEMM_BATCH
    const int group_count = 1;
    dgemm_batch(&op_a, &op_b, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, C, &LDC, &group_count, &batch_count);
#else // Fallback to loop
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, M, N, K, alpha, A[i], LDA, B[i], LDB, beta, C[i], LDC);
#endif
  }
  void gemm_batch(char op_a, char op_b, int M, int N, int K, std::complex<double> alpha, const std::complex<double> **A, int LDA,
                  const std::complex<double> **B, int LDB, std::complex<double> beta, std::complex<double> **C, int LDC, int batch_count) {
#ifdef HAVE_GEMM_BATCH
    const int group_count = 1;
    zgemm_batch(&op_a, &op_b, &M, &N, &K, blacplx(&alpha), blacplx(A), &LDA, blacplx(B), &LDB, blacplx(&beta), blacplx(C), &LDC, &group_count,
                &batch_count);
#else
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, M, N, K, alpha, A[i], LDA, B[i], LDB, beta, C[i], LDC);
#endif
  }

  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, int strideA, const double *B, int LDB,
                          int strideB, double beta, double *C, int LDC, int strideC, int batch_count) {
#ifdef HAVE_GEMM_BATCH
    dgemm_batch_strided(&op_a, &op_b, &M, &N, &K, &alpha, A, &LDA, &strideA, B, &LDB, &strideB, &beta, C, &LDC, &strideC, &batch_count);
#else
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, M, N, K, alpha, A + i * strideA, LDA, B + i * strideB, LDB, beta, C + i * strideC, LDC);
#endif
  }
  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, std::complex<double> alpha, const std::complex<double> *A, int LDA, int strideA,
                          const std::complex<double> *B, int LDB, int strideB, std::complex<double> beta, std::complex<double> *C, int LDC,
                          int strideC, int batch_count) {
#ifdef HAVE_GEMM_BATCH
    zgemm_batch_strided(&op_a, &op_b, &M, &N, &K, blacplx(&alpha), blacplx(A), &LDA, &strideA, blacplx(B), &LDB, &strideB, blacplx(&beta), blacplx(C),
                        &LDC, &strideC, &batch_count);
#else
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, M, N, K, alpha, A + i * strideA, LDA, B + i * strideB, LDB, beta, C + i * strideC, LDC);
#endif
  }

  void gemv(char op, int M, int N, double alpha, const double *A, int LDA, const double *x, int incx, double beta, double *Y, int incy) {
    F77_dgemv(&op, &M, &N, &alpha, A, &LDA, x, &incx, &beta, Y, &incy);
  }
  void gemv(char op, int M, int N, std::complex<double> alpha, const std::complex<double> *A, int LDA, const std::complex<double> *x, int incx,
            std::complex<double> beta, std::complex<double> *Y, int incy) {
    F77_zgemv(&op, &M, &N, blacplx(&alpha), blacplx(A), &LDA, blacplx(x), &incx, blacplx(&beta), blacplx(Y), &incy);
  }

  void ger(int M, int N, double alpha, const double *x, int incx, const double *Y, int incy, double *A, int LDA) {
    F77_dger(&M, &N, &alpha, x, &incx, Y, &incy, A, &LDA);
  }
  void ger(int M, int N, std::complex<double> alpha, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy,
           std::complex<double> *A, int LDA) {
    F77_zgeru(&M, &N, blacplx(&alpha), blacplx(x), &incx, blacplx(Y), &incy, blacplx(A), &LDA);
  }

  void scal(int M, double alpha, double *x, int incx) { F77_dscal(&M, &alpha, x, &incx); }
  void scal(int M, std::complex<double> alpha, std::complex<double> *x, int incx) { F77_zscal(&M, blacplx(&alpha), blacplx(x), &incx); }

  void swap(int N, double *x, int incx, double *Y, int incy) { F77_dswap(&N, x, &incx, Y, &incy); }
  void swap(int N, std::complex<double> *x, int incx, std::complex<double> *Y, int incy) { F77_zswap(&N, blacplx(x), &incx, blacplx(Y), &incy); }

} // namespace nda::blas::f77

namespace nda::blas::cuda {

  constexpr cublasOperation_t get_cublas_op(char op) {
    switch (op) {
      case 'N': return CUBLAS_OP_N; break;
      case 'T': return CUBLAS_OP_T; break;
      case 'C': return CUBLAS_OP_C; break;
      default: std::terminate(); return {};
    }
  }

  struct handle_t {
    handle_t() { cublasCreate(&h); }
    ~handle_t() { cublasDestroy(h); }
    operator cublasHandle_t() { return h; }

    private:
    cublasHandle_t h = {};
  };
  static handle_t handle = {};

  /// Global option to turn on/off the cudaDeviceSynchronize after cublas library calls
  static bool synchronize = true;
#define CUBLAS_CHECK(X, ...)                                                                                                                         \
  auto err = X(handle, __VA_ARGS__);                                                                                                                 \
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

  void axpy(int N, double alpha, const double *x, int incx, double *Y, int incy) { cublasDaxpy(handle, N, &alpha, x, incx, Y, incy); }
  void axpy(int N, std::complex<double> alpha, const std::complex<double> *x, int incx, std::complex<double> *Y, int incy) {
    CUBLAS_CHECK(cublasZaxpy, N, cucplx(&alpha), cucplx(x), incx, cucplx(Y), incy);
  }

  void copy(int N, const double *x, int incx, double *Y, int incy) { cublasDcopy(handle, N, x, incx, Y, incy); }
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

} // namespace nda::blas::cuda
