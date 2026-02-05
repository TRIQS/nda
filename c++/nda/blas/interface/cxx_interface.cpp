// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Implementation details for blas/interface/cxx_interface.hpp.
 */

// Extracted from Reference Lapack (https://github.com/Reference-LAPACK):
#include "./cblas_f77.h"
#include "./cxx_interface.hpp"
#include "../tools.hpp"

#include <cstddef>

#ifdef NDA_USE_MKL
#include "../../basic_array.hpp"
#include "../../declarations.hpp"

#include <mkl.h>

namespace nda::blas {

#ifdef NDA_USE_MKL_RT
  static int const mkl_interface_layer = mkl_set_interface_layer(MKL_INTERFACE_LP64 + MKL_INTERFACE_GNU);
#endif
  inline auto *mklcplx(std::complex<double> *c) { return reinterpret_cast<MKL_Complex16 *>(c); }               // NOLINT
  inline auto *mklcplx(std::complex<double> const *c) { return reinterpret_cast<const MKL_Complex16 *>(c); }   // NOLINT
  inline auto *mklcplx(std::complex<double> **c) { return reinterpret_cast<MKL_Complex16 **>(c); }             // NOLINT
  inline auto *mklcplx(std::complex<double> const **c) { return reinterpret_cast<const MKL_Complex16 **>(c); } // NOLINT

} // namespace nda::blas
#endif

namespace {

  // complex struct which is returned by BLAS functions
  struct nda_complex_double {
    double real;
    double imag;
  };

} // namespace

// manually define dot routines since cblas_f77.h uses "_sub" to wrap the Fortran routines
#define F77_ddot F77_GLOBAL(ddot, DDOT)
#define F77_zdotu F77_GLOBAL(zdotu, ZDOTU)
#define F77_zdotc F77_GLOBAL(zdotc, ZDOTC)
extern "C" {
double F77_ddot(FINT, const double *, FINT, const double *, FINT);
nda_complex_double F77_zdotu(FINT, const double *, FINT, const double *, FINT);
nda_complex_double F77_zdotc(FINT, const double *, FINT, const double *, FINT);
}

namespace nda::blas::f77 {

  inline auto *blacplx(std::complex<double> *c) { return reinterpret_cast<double *>(c); }                // NOLINT
  inline auto *blacplx(std::complex<double> const *c) { return reinterpret_cast<const double *>(c); }    // NOLINT
  inline auto **blacplx(std::complex<double> **c) { return reinterpret_cast<double **>(c); }             // NOLINT
  inline auto **blacplx(std::complex<double> const **c) { return reinterpret_cast<const double **>(c); } // NOLINT

  void axpy(int n, double alpha, const double *x, int incx, double *y, int incy) { F77_daxpy(&n, &alpha, x, &incx, y, &incy); }
  void axpy(int n, std::complex<double> alpha, const std::complex<double> *x, int incx, std::complex<double> *y, int incy) {
    F77_zaxpy(&n, blacplx(&alpha), blacplx(x), &incx, blacplx(y), &incy);
  }

  // No Const In Wrapping!
  void copy(int n, const double *x, int incx, double *y, int incy) { F77_dcopy(&n, x, &incx, y, &incy); }
  void copy(int n, const std::complex<double> *x, int incx, std::complex<double> *y, int incy) {
    F77_zcopy(&n, blacplx(x), &incx, blacplx(y), &incy);
  }

  double dot(int m, const double *x, int incx, const double *y, int incy) { return F77_ddot(&m, x, &incx, y, &incy); }
  std::complex<double> dot(int m, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy) {
#ifdef NDA_USE_MKL
    MKL_Complex16 result;
    cblas_zdotu_sub(m, mklcplx(x), incx, mklcplx(y), incy, &result);
#else
    auto result = F77_zdotu(&m, blacplx(x), &incx, blacplx(y), &incy);
#endif
    return std::complex<double>{result.real, result.imag};
  }
  std::complex<double> dotc(int m, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy) {
#ifdef NDA_USE_MKL
    MKL_Complex16 result;
    cblas_zdotc_sub(m, mklcplx(x), incx, mklcplx(y), incy, &result);
#else
    auto result = F77_zdotc(&m, blacplx(x), &incx, blacplx(y), &incy);
#endif
    return std::complex<double>{result.real, result.imag};
  }

  void gemm(char op_a, char op_b, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta, double *c,
            int ldc) {
    F77_dgemm(&op_a, &op_b, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  }
  void gemm(char op_a, char op_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda,
            const std::complex<double> *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc) {
    F77_zgemm(&op_a, &op_b, &m, &n, &k, blacplx(&alpha), blacplx(a), &lda, blacplx(b), &ldb, blacplx(&beta), blacplx(c), &ldc);
  }

  void gemm_batch(char op_a, char op_b, int m, int n, int k, double alpha, const double **a, int lda, const double **b, int ldb, double beta,
                  double **c, int ldc, int batch_count) {
#ifdef NDA_USE_MKL
    const int group_count = 1;
    dgemm_batch(&op_a, &op_b, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, &group_count, &batch_count);
#else // Fallback to loop
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, m, n, k, alpha, a[i], lda, b[i], ldb, beta, c[i], ldc);
#endif
  }
  void gemm_batch(char op_a, char op_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double> **a, int lda,
                  const std::complex<double> **b, int ldb, std::complex<double> beta, std::complex<double> **c, int ldc, int batch_count) {
#ifdef NDA_USE_MKL
    const int group_count = 1;
    zgemm_batch(&op_a, &op_b, &m, &n, &k, mklcplx(&alpha), mklcplx(a), &lda, mklcplx(b), &ldb, mklcplx(&beta), mklcplx(c), &ldc, &group_count,
                &batch_count);
#else
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, m, n, k, alpha, a[i], lda, b[i], ldb, beta, c[i], ldc);
#endif
  }

  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, double alpha, const double **a, int *lda, const double **b, int *ldb, double beta,
                   double **c, int *ldc, int batch_count) {
#ifdef NDA_USE_MKL
    nda::vector<int> group_size(batch_count, 1);
    nda::vector<char> ops_a(batch_count, op_a), ops_b(batch_count, op_b);
    nda::vector<double> alphas(batch_count, alpha), betas(batch_count, beta);
    dgemm_batch(ops_a.data(), ops_b.data(), m, n, k, alphas.data(), a, lda, b, ldb, betas.data(), c, ldc, &batch_count, group_size.data());
#else
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, m[i], n[i], k[i], alpha, a[i], lda[i], b[i], ldb[i], beta, c[i], ldc[i]);
#endif
  }
  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, std::complex<double> alpha, const std::complex<double> **a, int *lda,
                   const std::complex<double> **b, int *ldb, std::complex<double> beta, std::complex<double> **c, int *ldc, int batch_count) {
#ifdef NDA_USE_MKL
    nda::vector<int> group_size(batch_count, 1);
    nda::vector<char> ops_a(batch_count, op_a), ops_b(batch_count, op_b);
    nda::vector<std::complex<double>> alphas(batch_count, alpha), betas(batch_count, beta);
    zgemm_batch(ops_a.data(), ops_b.data(), m, n, k, mklcplx(alphas.data()), mklcplx(a), lda, mklcplx(b), ldb, mklcplx(betas.data()), mklcplx(c), ldc,
                &batch_count, group_size.data());
#else
    for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, m[i], n[i], k[i], alpha, a[i], lda[i], b[i], ldb[i], beta, c[i], ldc[i]);
#endif
  }

  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, double alpha, const double *a, int lda, int stride_a, const double *b, int ldb,
                          int stride_b, double beta, double *c, int ldc, int stride_c, int batch_count) {
#if defined(NDA_USE_MKL) && INTEL_MKL_VERSION >= 20200002
    dgemm_batch_strided(&op_a, &op_b, &m, &n, &k, &alpha, a, &lda, &stride_a, b, &ldb, &stride_b, &beta, c, &ldc, &stride_c, &batch_count);
#else
    for (int i = 0; i < batch_count; ++i)
      gemm(op_a, op_b, m, n, k, alpha, a + static_cast<ptrdiff_t>(i * stride_a), lda, b + static_cast<ptrdiff_t>(i * stride_b), ldb, beta,
           c + static_cast<ptrdiff_t>(i * stride_c), ldc);
#endif
  }
  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda, int stride_a,
                          const std::complex<double> *b, int ldb, int stride_b, std::complex<double> beta, std::complex<double> *c, int ldc,
                          int stride_c, int batch_count) {
#if defined(NDA_USE_MKL) && INTEL_MKL_VERSION >= 20200002
    zgemm_batch_strided(&op_a, &op_b, &m, &n, &k, mklcplx(&alpha), mklcplx(a), &lda, &stride_a, mklcplx(b), &ldb, &stride_b, mklcplx(&beta),
                        mklcplx(c), &ldc, &stride_c, &batch_count);
#else
    for (int i = 0; i < batch_count; ++i)
      gemm(op_a, op_b, m, n, k, alpha, a + static_cast<ptrdiff_t>(i * stride_a), lda, b + static_cast<ptrdiff_t>(i * stride_b), ldb, beta,
           c + static_cast<ptrdiff_t>(i * stride_c), ldc);
#endif
  }

  void gemv(char op, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy) {
    F77_dgemv(&op, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
  void gemv(char op, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda, const std::complex<double> *x, int incx,
            std::complex<double> beta, std::complex<double> *y, int incy) {
    F77_zgemv(&op, &m, &n, blacplx(&alpha), blacplx(a), &lda, blacplx(x), &incx, blacplx(&beta), blacplx(y), &incy);
  }

  void ger(int m, int n, double alpha, const double *x, int incx, const double *y, int incy, double *a, int lda) {
    F77_dger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  }
  void ger(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy,
           std::complex<double> *a, int lda) {
    F77_zgeru(&m, &n, blacplx(&alpha), blacplx(x), &incx, blacplx(y), &incy, blacplx(a), &lda);
  }
  void gerc(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy,
            std::complex<double> *a, int lda) {
    F77_zgerc(&m, &n, blacplx(&alpha), blacplx(x), &incx, blacplx(y), &incy, blacplx(a), &lda);
  }

  void scal(int m, double alpha, double *x, int incx) { F77_dscal(&m, &alpha, x, &incx); }
  void scal(int m, std::complex<double> alpha, std::complex<double> *x, int incx) { F77_zscal(&m, blacplx(&alpha), blacplx(x), &incx); }

  void swap(int n, double *x, int incx, double *y, int incy) { F77_dswap(&n, x, &incx, y, &incy); } // NOLINT (this is a BLAS swap)
  void swap(int n, std::complex<double> *x, int incx, std::complex<double> *y, int incy) {          // NOLINT (this is a BLAS swap)
    F77_zswap(&n, blacplx(x), &incx, blacplx(y), &incy);
  }

} // namespace nda::blas::f77
