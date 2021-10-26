// Copyright (c) 2019 Simons Foundation
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

#include "cxx_interface.hpp"
// Extracted from Reference Lapack (https://github.com/Reference-LAPACK):
#include "cblas_f77.h"

// Manual Include since cblas uses Fortan _sub to wrap function again
#define F77_ddot F77_GLOBAL(ddot, DDOT) //TRIQS Manual
extern "C" {
double F77_ddot(FINT, const double *, FINT, const double *, FINT);
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

  void gemm(char trans_a, char trans_b, int M, int N, int K, double alpha, const double *A, int LDA, const double *B, int LDB, double beta, double *C,
            int LDC) {
    F77_dgemm(&trans_a, &trans_b, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, C, &LDC);
  }
  void gemm(char trans_a, char trans_b, int M, int N, int K, std::complex<double> alpha, const std::complex<double> *A, int LDA,
            const std::complex<double> *B, int LDB, std::complex<double> beta, std::complex<double> *C, int LDC) {
    F77_zgemm(&trans_a, &trans_b, &M, &N, &K, reinterpret_cast<const double *>(&alpha), reinterpret_cast<const double *>(A), &LDA,      // NOLINT
              reinterpret_cast<const double *>(B), &LDB, reinterpret_cast<const double *>(&beta), reinterpret_cast<double *>(C), &LDC); // NOLINT
  }

  void gemv(char trans, int M, int N, double alpha, const double *A, int &LDA, const double *x, int incx, double beta, double *Y, int incy) {
    F77_dgemv(&trans, &M, &N, &alpha, A, &LDA, x, &incx, &beta, Y, &incy);
  }
  void gemv(char trans, int M, int N, std::complex<double> alpha, const std::complex<double> *A, int &LDA, const std::complex<double> *x, int incx,
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
