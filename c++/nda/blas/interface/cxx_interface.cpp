// Copyright (c) 2019-2022 Simons Foundation
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
// Authors: Miguel Morales, Olivier Parcollet, Nils Wentzell

#include <nda/nda.hpp>
#include <nda/exceptions.hpp>
#include "cxx_interface.hpp"

// Extracted from Reference Lapack (https://github.com/Reference-LAPACK):
#include "cblas_f77.h"

#include <string>
#include <vector>

using namespace std::string_literals;

#ifdef NDA_USE_MKL
#include <mkl.h>
namespace nda::blas {
  static int const mkl_interface_layer = mkl_set_interface_layer(MKL_INTERFACE_LP64 + MKL_INTERFACE_GNU);
  inline auto *mklcplx(nda::dcomplex *c) { return reinterpret_cast<MKL_Complex16 *>(c); }               // NOLINT
  inline auto *mklcplx(nda::dcomplex const *c) { return reinterpret_cast<const MKL_Complex16 *>(c); }   // NOLINT
  inline auto *mklcplx(nda::dcomplex **c) { return reinterpret_cast<MKL_Complex16 **>(c); }             // NOLINT
  inline auto *mklcplx(nda::dcomplex const **c) { return reinterpret_cast<const MKL_Complex16 **>(c); } // NOLINT
} // namespace nda::blas
#endif

// Define a complex struct which is returned by BLAS functions
namespace {
  struct nda_complex_double {
    double real;
    double imag;
  };
} // namespace

// Manual Include since cblas uses Fortan _sub to wrap function again
#define F77_ddot F77_GLOBAL(ddot, DDOT)
#define F77_zdotu F77_GLOBAL(zdotu, DDOT)
#define F77_zdotc F77_GLOBAL(zdotc, DDOT)
extern "C" {
double F77_ddot(FINT, const double *, FINT, const double *, FINT);
nda_complex_double F77_zdotu(FINT, const double *, FINT, const double *, FINT); // NOLINT
nda_complex_double F77_zdotc(FINT, const double *, FINT, const double *, FINT); // NOLINT
}

namespace nda::blas::f77 {

  inline auto *blacplx(dcomplex *c) { return reinterpret_cast<double *>(c); }                // NOLINT
  inline auto *blacplx(dcomplex const *c) { return reinterpret_cast<const double *>(c); }    // NOLINT
  inline auto **blacplx(dcomplex **c) { return reinterpret_cast<double **>(c); }             // NOLINT
  inline auto **blacplx(dcomplex const **c) { return reinterpret_cast<const double **>(c); } // NOLINT

  void axpy(int N, double alpha, const double *x, int incx, double *Y, int incy) { F77_daxpy(&N, &alpha, x, &incx, Y, &incy); }
  void axpy(int N, dcomplex alpha, const dcomplex *x, int incx, dcomplex *Y, int incy) {
    F77_zaxpy(&N, blacplx(&alpha), blacplx(x), &incx, blacplx(Y), &incy);
  }
  // No Const In Wrapping!
  void copy(int N, const double *x, int incx, double *Y, int incy) { F77_dcopy(&N, x, &incx, Y, &incy); }
  void copy(int N, const dcomplex *x, int incx, dcomplex *Y, int incy) { F77_zcopy(&N, blacplx(x), &incx, blacplx(Y), &incy); }

  double dot(int M, const double *x, int incx, const double *Y, int incy) { return F77_ddot(&M, x, &incx, Y, &incy); }
  dcomplex dot(int M, const dcomplex *x, int incx, const dcomplex *Y, int incy) {
    auto result = F77_zdotu(&M, blacplx(x), &incx, blacplx(Y), &incy);
    return dcomplex{result.real, result.imag};
  }
  dcomplex dotc(int M, const dcomplex *x, int incx, const dcomplex *Y, int incy) {
    auto result = F77_zdotc(&M, blacplx(x), &incx, blacplx(Y), &incy);
    return dcomplex{result.real, result.imag};
  }

  void gemm(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, const double *B, int LDB, double beta, double *C,
            int LDC) {
    F77_dgemm(&op_a, &op_b, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, C, &LDC);
  }
  void gemm(char op_a, char op_b, int M, int N, int K, dcomplex alpha, const dcomplex *A, int LDA, const dcomplex *B, int LDB, dcomplex beta,
            dcomplex *C, int LDC) {
    F77_zgemm(&op_a, &op_b, &M, &N, &K, blacplx(&alpha), blacplx(A), &LDA, blacplx(B), &LDB, blacplx(&beta), blacplx(C), &LDC);
  }

  void gemm_batch(char op_a, char op_b, int M, int N, int K, double alpha, const double **A, int LDA, const double **B, int LDB, double beta,
                  double **C, int LDC, int batch_count) {
#ifdef NDA_USE_MKL
    const int group_count = 1;
    dgemm_batch(&op_a, &op_b, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, C, &LDC, &group_count, &batch_count);
#else // Fallback to loop
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, M, N, K, alpha, A[i], LDA, B[i], LDB, beta, C[i], LDC);
#endif
  }
  void gemm_batch(char op_a, char op_b, int M, int N, int K, dcomplex alpha, const dcomplex **A, int LDA, const dcomplex **B, int LDB, dcomplex beta,
                  dcomplex **C, int LDC, int batch_count) {
#ifdef NDA_USE_MKL
    const int group_count = 1;
    zgemm_batch(&op_a, &op_b, &M, &N, &K, mklcplx(&alpha), mklcplx(A), &LDA, mklcplx(B), &LDB, mklcplx(&beta), mklcplx(C), &LDC, &group_count,
                &batch_count);
#else
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, M, N, K, alpha, A[i], LDA, B[i], LDB, beta, C[i], LDC);
#endif
  }

  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, double alpha, const double **A, int *LDA, const double **B, int *LDB, double beta,
                   double **C, int *LDC, int batch_count) {
#ifdef NDA_USE_MKL
    nda::vector<int> group_size(batch_count, 1);
    nda::vector<char> ops_a(batch_count, op_a), ops_b(batch_count, op_b);
    nda::vector<double> alphas(batch_count, alpha), betas(batch_count, beta);
    dgemm_batch(ops_a.data(), ops_b.data(), M, N, K, alphas.data(), A, LDA, B, LDB, betas.data(), C, LDC, &batch_count, group_size.data());
#else
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, M[i], N[i], K[i], alpha, A[i], LDA[i], B[i], LDB[i], beta, C[i], LDC[i]);
#endif
  }
  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, dcomplex alpha, const dcomplex **A, int *LDA, const dcomplex **B, int *LDB,
                   dcomplex beta, dcomplex **C, int *LDC, int batch_count) {
#ifdef NDA_USE_MKL
    nda::vector<int> group_size(batch_count, 1);
    nda::vector<char> ops_a(batch_count, op_a), ops_b(batch_count, op_b);
    nda::vector<dcomplex> alphas(batch_count, alpha), betas(batch_count, beta);
    zgemm_batch(ops_a.data(), ops_b.data(), M, N, K, mklcplx(alphas.data()), mklcplx(A), LDA, mklcplx(B), LDB, mklcplx(betas.data()), mklcplx(C), LDC,
                &batch_count, group_size.data());
#else
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, M[i], N[i], K[i], alpha, A[i], LDA[i], B[i], LDB[i], beta, C[i], LDC[i]);
#endif
  }

  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, int strideA, const double *B, int LDB,
                          int strideB, double beta, double *C, int LDC, int strideC, int batch_count) {
#ifdef NDA_USE_MKL
    dgemm_batch_strided(&op_a, &op_b, &M, &N, &K, &alpha, A, &LDA, &strideA, B, &LDB, &strideB, &beta, C, &LDC, &strideC, &batch_count);
#else
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, M, N, K, alpha, A + i * strideA, LDA, B + i * strideB, LDB, beta, C + i * strideC, LDC);
#endif
  }
  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, dcomplex alpha, const dcomplex *A, int LDA, int strideA, const dcomplex *B,
                          int LDB, int strideB, dcomplex beta, dcomplex *C, int LDC, int strideC, int batch_count) {
#ifdef NDA_USE_MKL
    zgemm_batch_strided(&op_a, &op_b, &M, &N, &K, mklcplx(&alpha), mklcplx(A), &LDA, &strideA, mklcplx(B), &LDB, &strideB, mklcplx(&beta), mklcplx(C),
                        &LDC, &strideC, &batch_count);
#else
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, M, N, K, alpha, A + i * strideA, LDA, B + i * strideB, LDB, beta, C + i * strideC, LDC);
#endif
  }

  void gemv(char op, int M, int N, double alpha, const double *A, int LDA, const double *x, int incx, double beta, double *Y, int incy) {
    F77_dgemv(&op, &M, &N, &alpha, A, &LDA, x, &incx, &beta, Y, &incy);
  }
  void gemv(char op, int M, int N, dcomplex alpha, const dcomplex *A, int LDA, const dcomplex *x, int incx, dcomplex beta, dcomplex *Y, int incy) {
    F77_zgemv(&op, &M, &N, blacplx(&alpha), blacplx(A), &LDA, blacplx(x), &incx, blacplx(&beta), blacplx(Y), &incy);
  }

  void ger(int M, int N, double alpha, const double *x, int incx, const double *Y, int incy, double *A, int LDA) {
    F77_dger(&M, &N, &alpha, x, &incx, Y, &incy, A, &LDA);
  }
  void ger(int M, int N, dcomplex alpha, const dcomplex *x, int incx, const dcomplex *Y, int incy, dcomplex *A, int LDA) {
    F77_zgeru(&M, &N, blacplx(&alpha), blacplx(x), &incx, blacplx(Y), &incy, blacplx(A), &LDA);
  }

  void scal(int M, double alpha, double *x, int incx) { F77_dscal(&M, &alpha, x, &incx); }
  void scal(int M, dcomplex alpha, dcomplex *x, int incx) { F77_zscal(&M, blacplx(&alpha), blacplx(x), &incx); }

  void swap(int N, double *x, int incx, double *Y, int incy) { F77_dswap(&N, x, &incx, Y, &incy); }
  void swap(int N, dcomplex *x, int incx, dcomplex *Y, int incy) { F77_zswap(&N, blacplx(x), &incx, blacplx(Y), &incy); }

} // namespace nda::blas::f77
