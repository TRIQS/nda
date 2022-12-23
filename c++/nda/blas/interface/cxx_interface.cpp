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
#include "cblas_f77.h"

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
#ifdef NDA_HAVE_GEMM_BATCH
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
#ifdef NDA_HAVE_GEMM_BATCH
    const int group_count = 1;
    dgemm_batch(&op_a, &op_b, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, C, &LDC, &group_count, &batch_count);
#else // Fallback to loop
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, M, N, K, alpha, A[i], LDA, B[i], LDB, beta, C[i], LDC);
#endif
  }
  void gemm_batch(char op_a, char op_b, int M, int N, int K, std::complex<double> alpha, const std::complex<double> **A, int LDA,
                  const std::complex<double> **B, int LDB, std::complex<double> beta, std::complex<double> **C, int LDC, int batch_count) {
#ifdef NDA_HAVE_GEMM_BATCH
    const int group_count = 1;
    zgemm_batch(&op_a, &op_b, &M, &N, &K, blacplx(&alpha), blacplx(A), &LDA, blacplx(B), &LDB, blacplx(&beta), blacplx(C), &LDC, &group_count,
                &batch_count);
#else
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, M, N, K, alpha, A[i], LDA, B[i], LDB, beta, C[i], LDC);
#endif
  }

  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, double alpha, const double **A, int *LDA, const double **B, int *LDB, double beta,
                   double **C, int *LDC, int batch_count) {
#ifdef NDA_HAVE_GEMM_BATCH
    nda::vector<int> group_size(batch_count, 1); 
    nda::vector<char> ops_a(batch_count, op_a), ops_b(batch_count, op_b);
    nda::vector<double> alphas(batch_count, alpha), betas(batch_count, beta);
    dgemm_batch(ops_a.data(), ops_b.data(), M, N, K, alphas.data(), A, LDA, B, LDB, betas.data(), C, LDC, &batch_count, group_size.data());
#else
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, M[i], N[i], K[i], alpha, A[i], LDA[i], B[i], LDB[i], beta, C[i], LDC[i]);
#endif
  }
  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, std::complex<double> alpha, const std::complex<double> **A, int *LDA,
                   const std::complex<double> **B, int *LDB, std::complex<double> beta, std::complex<double> **C, int *LDC, int batch_count) {
#ifdef NDA_HAVE_GEMM_BATCH
    nda::vector<int> group_size(batch_count, 1); 
    nda::vector<char> ops_a(batch_count, op_a), ops_b(batch_count, op_b);
    nda::vector<std::complex<double>> alphas(batch_count, alpha), betas(batch_count, beta);
    zgemm_batch(ops_a.data(), ops_b.data(), M, N, K, blacplx(alphas.data()), blacplx(A), LDA, blacplx(B), LDB, blacplx(betas.data()), blacplx(C), LDC,
                &batch_count, group_size.data());
#else
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, M[i], N[i], K[i], alpha, A[i], LDA[i], B[i], LDB[i], beta, C[i], LDC[i]);
#endif
  }

  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, int strideA, const double *B, int LDB,
                          int strideB, double beta, double *C, int LDC, int strideC, int batch_count) {
#ifdef NDA_HAVE_GEMM_BATCH
    dgemm_batch_strided(&op_a, &op_b, &M, &N, &K, &alpha, A, &LDA, &strideA, B, &LDB, &strideB, &beta, C, &LDC, &strideC, &batch_count);
#else
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, M, N, K, alpha, A + i * strideA, LDA, B + i * strideB, LDB, beta, C + i * strideC, LDC);
#endif
  }
  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, std::complex<double> alpha, const std::complex<double> *A, int LDA, int strideA,
                          const std::complex<double> *B, int LDB, int strideB, std::complex<double> beta, std::complex<double> *C, int LDC,
                          int strideC, int batch_count) {
#ifdef NDA_HAVE_GEMM_BATCH
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

