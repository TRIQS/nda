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
#include "cxx_interface.hpp"

// Extracted from Reference Lapack (https://github.com/Reference-LAPACK):
#include "cblas_f77.h"
#include "cublas_v2.h"

#include <string>

using namespace std::string_literals;

// Manual Include since cblas uses Fortan _sub to wrap function again
#define F77_ddot F77_GLOBAL(ddot, DDOT)
#define F77_zdotu F77_GLOBAL(zdotu, DDOT)
#define F77_zdotc F77_GLOBAL(zdotc, DDOT)
extern "C" {
double F77_ddot(FINT, const double *, FINT, const double *, FINT);
std::complex<double> F77_zdotu(FINT, const double *, FINT, const double *, FINT); // NOLINT
std::complex<double> F77_zdotc(FINT, const double *, FINT, const double *, FINT); // NOLINT
}

namespace nda::blas::f77 {

  void axpy(int N, double alpha, const double *x, int incx, double *Y, int incy) { F77_daxpy(&N, &alpha, x, &incx, Y, &incy); }
  void axpy(int N, std::complex<double> alpha, const std::complex<double> *x, int incx, std::complex<double> *Y, int incy) {
    F77_zaxpy(&N, reinterpret_cast<const double *>(&alpha), reinterpret_cast<const double *>(x), &incx, reinterpret_cast<double *>(Y), // NOLINT
              &incy);                                                                                                                  // NOLINT
  }
  // No Const In Wrapping!
  void copy(int N, const double *x, int incx, double *Y, int incy) { F77_dcopy(&N, x, &incx, Y, &incy); }
  void copy(int N, const std::complex<double> *x, int incx, std::complex<double> *Y, int incy) {
    F77_zcopy(&N, reinterpret_cast<const double *>(x), &incx, reinterpret_cast<double *>(Y), &incy); // NOLINT
  }

  double dot(int M, const double *x, int incx, const double *Y, int incy) { return F77_ddot(&M, x, &incx, Y, &incy); }
  std::complex<double> dot(int M, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy) {
    return F77_zdotu(&M, reinterpret_cast<const double *>(x), &incx, reinterpret_cast<const double *>(Y), &incy); // NOLINT
  }
  std::complex<double> dotc(int M, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy) {
    return F77_zdotc(&M, reinterpret_cast<const double *>(x), &incx, reinterpret_cast<const double *>(Y), &incy); // NOLINT
  }

  void gemm(char trans_a, char trans_b, int M, int N, int K, double alpha, const double *A, int LDA, const double *B, int LDB, double beta, double *C,
            int LDC) {
    F77_dgemm(&trans_a, &trans_b, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, C, &LDC);
  }
  void gemm(char trans_a, char trans_b, int M, int N, int K, std::complex<double> alpha, const std::complex<double> *A, int LDA,
            const std::complex<double> *B, int LDB, std::complex<double> beta, std::complex<double> *C, int LDC) {
    F77_zgemm(&trans_a, &trans_b, &M, &N, &K, reinterpret_cast<const double *>(&alpha), reinterpret_cast<const double *>(A), &LDA,      // NOLINT
              reinterpret_cast<const double *>(B), &LDB, reinterpret_cast<const double *>(&beta), reinterpret_cast<double *>(C), &LDC); // NOLINT
  }

  void gemv(char trans, int M, int N, double alpha, const double *A, int LDA, const double *x, int incx, double beta, double *Y, int incy) {
    F77_dgemv(&trans, &M, &N, &alpha, A, &LDA, x, &incx, &beta, Y, &incy);
  }
  void gemv(char trans, int M, int N, std::complex<double> alpha, const std::complex<double> *A, int LDA, const std::complex<double> *x, int incx,
            std::complex<double> beta, std::complex<double> *Y, int incy) {
    F77_zgemv(&trans, &M, &N, reinterpret_cast<const double *>(&alpha), reinterpret_cast<const double *>(A), &LDA,                        // NOLINT
              reinterpret_cast<const double *>(x), &incx, reinterpret_cast<const double *>(&beta), reinterpret_cast<double *>(Y), &incy); // NOLINT
  }

  void ger(int M, int N, double alpha, const double *x, int incx, const double *Y, int incy, double *A, int LDA) {
    F77_dger(&M, &N, &alpha, x, &incx, Y, &incy, A, &LDA);
  }
  void ger(int M, int N, std::complex<double> alpha, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy,
           std::complex<double> *A, int LDA) {
    F77_zgeru(&M, &N, reinterpret_cast<const double *>(&alpha), reinterpret_cast<const double *>(x), &incx, // NOLINT
              reinterpret_cast<const double *>(Y),                                                          // NOLINT
              &incy, reinterpret_cast<double *>(A), &LDA);                                                  // NOLINT
  }

  void scal(int M, double alpha, double *x, int incx) { F77_dscal(&M, &alpha, x, &incx); }
  void scal(int M, std::complex<double> alpha, std::complex<double> *x, int incx) {
    F77_zscal(&M, reinterpret_cast<const double *>(&alpha), reinterpret_cast<double *>(x), &incx); // NOLINT
  }

  void swap(int N, double *x, int incx, double *Y, int incy) { F77_dswap(&N, x, &incx, Y, &incy); }
  void swap(int N, std::complex<double> *x, int incx, std::complex<double> *Y, int incy) {
    F77_zswap(&N, reinterpret_cast<double *>(x), &incx, reinterpret_cast<double *>(Y), &incy); // NOLINT
  }

} // namespace nda::blas::f77

namespace nda::blas::cuda {

  constexpr cublasOperation_t get_op(char trans) {
    switch (trans) {
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

#define CUBLAS_CHECK(X, ...)                                                                                                                         \
  auto err = X(handle, __VA_ARGS__);                                                                                                                 \
  ASSERT_WITH_MESSAGE(err == CUBLAS_STATUS_SUCCESS, AS_STRING(X) + " failed with error code "s + std::to_string(err));

  void gemm(char trans_a, char trans_b, int M, int N, int K, double alpha, const double *A, int LDA, const double *B, int LDB, double beta, double *C,
            int LDC) {
    CUBLAS_CHECK(cublasDgemm, get_op(trans_a), get_op(trans_b), M, N, K, &alpha, A, LDA, B, LDB, &beta, C, LDC); // NOLINT
  }

  void gemm(char trans_a, char trans_b, int M, int N, int K, std::complex<double> alpha, const std::complex<double> *A, int LDA,
            const std::complex<double> *B, int LDB, std::complex<double> beta, std::complex<double> *C, int LDC) {
    auto alpha_cu = cuDoubleComplex{alpha.real(), alpha.imag()};
    auto beta_cu  = cuDoubleComplex{beta.real(), beta.imag()};
    CUBLAS_CHECK(cublasZgemm, get_op(trans_a), get_op(trans_b), M, N, K, &alpha_cu, reinterpret_cast<const cuDoubleComplex *>(A), LDA, // NOLINT
                 reinterpret_cast<const cuDoubleComplex *>(B), LDB, &beta_cu, reinterpret_cast<cuDoubleComplex *>(C), LDC);            // NOLINT
  }

  void axpy(int N, double alpha, const double *x, int incx, double *Y, int incy) { cublasDaxpy(handle, N, &alpha, x, incx, Y, incy); }
  void axpy(int N, std::complex<double> alpha, const std::complex<double> *x, int incx, std::complex<double> *Y, int incy) {
    CUBLAS_CHECK(cublasZaxpy, N, reinterpret_cast<const cuDoubleComplex *>(&alpha), reinterpret_cast<const cuDoubleComplex *>(x), incx, // NOLINT
                 reinterpret_cast<cuDoubleComplex *>(Y), incy);                                                                         // NOLINT
  }

  void copy(int N, const double *x, int incx, double *Y, int incy) { cublasDcopy(handle, N, x, incx, Y, incy); }
  void copy(int N, const std::complex<double> *x, int incx, std::complex<double> *Y, int incy) {
    CUBLAS_CHECK(cublasZcopy, N, reinterpret_cast<const cuDoubleComplex *>(x), incx, reinterpret_cast<cuDoubleComplex *>(Y), incy); // NOLINT
  }

  double dot(int M, const double *x, int incx, const double *Y, int incy) {
    double res{};
    CUBLAS_CHECK(cublasDdot, M, x, incx, Y, incy, &res); // NOLINT
    return res;
  }
  std::complex<double> dot(int M, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy) {
    cuDoubleComplex res;
    CUBLAS_CHECK(cublasZdotu, M, reinterpret_cast<const cuDoubleComplex *>(x), incx, reinterpret_cast<const cuDoubleComplex *>(Y), incy,
                 &res); // NOLINT
    return {res.x, res.y};
  }
  std::complex<double> dotc(int M, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy) {
    cuDoubleComplex res;
    CUBLAS_CHECK(cublasZdotc, M, reinterpret_cast<const cuDoubleComplex *>(x), incx, reinterpret_cast<const cuDoubleComplex *>(Y), incy,
                 &res); // NOLINT
    return {res.x, res.y};
  }

  void gemv(char trans, int M, int N, double alpha, const double *A, int LDA, const double *x, int incx, double beta, double *Y, int incy) {
    CUBLAS_CHECK(cublasDgemv, get_op(trans), M, N, &alpha, A, LDA, x, incx, &beta, Y, incy); // NOLINT
  }
  void gemv(char trans, int M, int N, std::complex<double> alpha, const std::complex<double> *A, int LDA, const std::complex<double> *x, int incx,
            std::complex<double> beta, std::complex<double> *Y, int incy) {
    CUBLAS_CHECK(cublasZgemv, get_op(trans), M, N, reinterpret_cast<const cuDoubleComplex *>(&alpha),
                 reinterpret_cast<const cuDoubleComplex *>(A),                                                              // NOLINT
                 LDA, reinterpret_cast<const cuDoubleComplex *>(x), incx, reinterpret_cast<const cuDoubleComplex *>(&beta), // NOLINT
                 reinterpret_cast<cuDoubleComplex *>(Y), incy);                                                             // NOLINT
  }

  void ger(int M, int N, double alpha, const double *x, int incx, const double *Y, int incy, double *A, int LDA) {
    CUBLAS_CHECK(cublasDger, M, N, &alpha, x, incx, Y, incy, A, LDA); // NOLINT
  }
  void ger(int M, int N, std::complex<double> alpha, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy,
           std::complex<double> *A, int LDA) {
    CUBLAS_CHECK(cublasZgeru, M, N, reinterpret_cast<const cuDoubleComplex *>(&alpha), reinterpret_cast<const cuDoubleComplex *>(x), incx, // NOLINT
                 reinterpret_cast<const cuDoubleComplex *>(Y), incy, reinterpret_cast<cuDoubleComplex *>(A), LDA);                         // NOLINT
  }

  void scal(int M, double alpha, double *x, int incx) { CUBLAS_CHECK(cublasDscal, M, &alpha, x, incx); } // NOLINT
  void scal(int M, std::complex<double> alpha, std::complex<double> *x, int incx) {
    CUBLAS_CHECK(cublasZscal, M, reinterpret_cast<const cuDoubleComplex *>(&alpha), reinterpret_cast<cuDoubleComplex *>(x), incx); // NOLINT
  }

  void swap(int N, double *x, int incx, double *Y, int incy) { CUBLAS_CHECK(cublasDswap, N, x, incx, Y, incy); } // NOLINT
  void swap(int N, std::complex<double> *x, int incx, std::complex<double> *Y, int incy) {
    CUBLAS_CHECK(cublasZswap, N, reinterpret_cast<cuDoubleComplex *>(x), incx, reinterpret_cast<cuDoubleComplex *>(Y), incy); // NOLINT
  }

} // namespace nda::blas::cuda
