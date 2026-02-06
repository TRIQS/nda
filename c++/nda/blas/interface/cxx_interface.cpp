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
  inline auto *mklcplx(std::complex<float> *c) { return reinterpret_cast<MKL_Complex8 *>(c); }               // NOLINT
  inline auto *mklcplx(std::complex<float> const *c) { return reinterpret_cast<const MKL_Complex8 *>(c); }   // NOLINT
  inline auto *mklcplx(std::complex<float> **c) { return reinterpret_cast<MKL_Complex8 **>(c); }             // NOLINT
  inline auto *mklcplx(std::complex<float> const **c) { return reinterpret_cast<const MKL_Complex8 **>(c); } // NOLINT

  inline auto *mklcplx(std::complex<double> *c) { return reinterpret_cast<MKL_Complex16 *>(c); }               // NOLINT
  inline auto *mklcplx(std::complex<double> const *c) { return reinterpret_cast<const MKL_Complex16 *>(c); }   // NOLINT
  inline auto *mklcplx(std::complex<double> **c) { return reinterpret_cast<MKL_Complex16 **>(c); }             // NOLINT
  inline auto *mklcplx(std::complex<double> const **c) { return reinterpret_cast<const MKL_Complex16 **>(c); } // NOLINT

} // namespace nda::blas
#endif

namespace {

  // single-precision complex struct which is returned by BLAS functions
  struct nda_complex_float {
    float real;
    float imag;
  };

  // double-precision complex struct which is returned by BLAS functions
  struct nda_complex_double {
    double real;
    double imag;
  };

} // namespace

// manually define dot routines since cblas_f77.h uses "_sub" to wrap the Fortran routines
#define F77_sdot F77_GLOBAL(sdot, SDOT)
#define F77_cdotu F77_GLOBAL(cdotu, CDOTU)
#define F77_cdotc F77_GLOBAL(cdotc, CDOTC)
#define F77_ddot F77_GLOBAL(ddot, DDOT)
#define F77_zdotu F77_GLOBAL(zdotu, ZDOTU)
#define F77_zdotc F77_GLOBAL(zdotc, ZDOTC)

extern "C" {
float F77_sdot(FINT, const float *, FINT, const float *, FINT);
nda_complex_float F77_cdotu(FINT, const float *, FINT, const float *, FINT);
nda_complex_float F77_cdotc(FINT, const float *, FINT, const float *, FINT);

double F77_ddot(FINT, const double *, FINT, const double *, FINT);
nda_complex_double F77_zdotu(FINT, const double *, FINT, const double *, FINT);
nda_complex_double F77_zdotc(FINT, const double *, FINT, const double *, FINT);
}

namespace nda::blas::f77 {

  inline auto *blacplx(std::complex<float> *c) { return reinterpret_cast<float *>(c); }                // NOLINT
  inline auto *blacplx(std::complex<float> const *c) { return reinterpret_cast<const float *>(c); }    // NOLINT
  inline auto **blacplx(std::complex<float> **c) { return reinterpret_cast<float **>(c); }             // NOLINT
  inline auto **blacplx(std::complex<float> const **c) { return reinterpret_cast<const float **>(c); } // NOLINT

  inline auto *blacplx(std::complex<double> *c) { return reinterpret_cast<double *>(c); }                // NOLINT
  inline auto *blacplx(std::complex<double> const *c) { return reinterpret_cast<const double *>(c); }    // NOLINT
  inline auto **blacplx(std::complex<double> **c) { return reinterpret_cast<double **>(c); }             // NOLINT
  inline auto **blacplx(std::complex<double> const **c) { return reinterpret_cast<const double **>(c); } // NOLINT

  namespace {

    // Helper function to call gemm_batch routine.
    template <typename T>
    void gemm_batch_impl(char op_a, char op_b, int m, int n, int k, T alpha, const T **a, int lda, const T **b, int ldb, T beta, T **c, int ldc,
                         int batch_count) {
#ifdef NDA_USE_MKL
      const int group_count = 1;
      if constexpr (std::is_same_v<T, float>) {
        sgemm_batch(&op_a, &op_b, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, &group_count, &batch_count);
      } else if constexpr (std::is_same_v<T, double>) {
        dgemm_batch(&op_a, &op_b, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, &group_count, &batch_count);
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        cgemm_batch(&op_a, &op_b, &m, &n, &k, mklcplx(&alpha), mklcplx(a), &lda, mklcplx(b), &ldb, mklcplx(&beta), mklcplx(c), &ldc, &group_count,
                    &batch_count);
      } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        zgemm_batch(&op_a, &op_b, &m, &n, &k, mklcplx(&alpha), mklcplx(a), &lda, mklcplx(b), &ldb, mklcplx(&beta), mklcplx(c), &ldc, &group_count,
                    &batch_count);
      }
#else
      for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, m, n, k, alpha, a[i], lda, b[i], ldb, beta, c[i], ldc);
#endif
    }

    // Helper function to call gemm_batch routine with variable matrix sizes.
    template <typename T>
    void gemm_vbatch_impl(char op_a, char op_b, int *m, int *n, int *k, T alpha, const T **a, int *lda, const T **b, int *ldb, T beta, T **c,
                          int *ldc, int batch_count) {
#ifdef NDA_USE_MKL
      nda::vector<int> group_size(batch_count, 1);
      nda::vector<char> ops_a(batch_count, op_a), ops_b(batch_count, op_b);
      nda::vector<T> alphas(batch_count, alpha), betas(batch_count, beta);
      if constexpr (std::is_same_v<T, float>) {
        sgemm_batch(ops_a.data(), ops_b.data(), m, n, k, alphas.data(), a, lda, b, ldb, betas.data(), c, ldc, &batch_count, group_size.data());
      } else if constexpr (std::is_same_v<T, double>) {
        dgemm_batch(ops_a.data(), ops_b.data(), m, n, k, alphas.data(), a, lda, b, ldb, betas.data(), c, ldc, &batch_count, group_size.data());
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        cgemm_batch(ops_a.data(), ops_b.data(), m, n, k, mklcplx(alphas.data()), mklcplx(a), lda, mklcplx(b), ldb, mklcplx(betas.data()), mklcplx(c),
                    ldc, &batch_count, group_size.data());
      } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        zgemm_batch(ops_a.data(), ops_b.data(), m, n, k, mklcplx(alphas.data()), mklcplx(a), lda, mklcplx(b), ldb, mklcplx(betas.data()), mklcplx(c),
                    ldc, &batch_count, group_size.data());
      }
#else
      for (int i = 0; i < batch_count; ++i) gemm(op_a, op_b, m[i], n[i], k[i], alpha, a[i], lda[i], b[i], ldb[i], beta, c[i], ldc[i]);
#endif
    }

    // Helper function to call gemm_batch_strided routine.
    template <typename T>
    void gemm_batch_strided_impl(char op_a, char op_b, int m, int n, int k, T alpha, const T *a, int lda, int stride_a, const T *b, int ldb,
                                 int stride_b, T beta, T *c, int ldc, int stride_c, int batch_count) {
#if defined(NDA_USE_MKL) && INTEL_MKL_VERSION >= 20200002
      if constexpr (std::is_same_v<T, float>) {
        sgemm_batch_strided(&op_a, &op_b, &m, &n, &k, &alpha, a, &lda, &stride_a, b, &ldb, &stride_b, &beta, c, &ldc, &stride_c, &batch_count);
      } else if constexpr (std::is_same_v<T, double>) {
        dgemm_batch_strided(&op_a, &op_b, &m, &n, &k, &alpha, a, &lda, &stride_a, b, &ldb, &stride_b, &beta, c, &ldc, &stride_c, &batch_count);
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        cgemm_batch_strided(&op_a, &op_b, &m, &n, &k, mklcplx(&alpha), mklcplx(a), &lda, &stride_a, mklcplx(b), &ldb, &stride_b, mklcplx(&beta),
                            mklcplx(c), &ldc, &stride_c, &batch_count);
      } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        zgemm_batch_strided(&op_a, &op_b, &m, &n, &k, mklcplx(&alpha), mklcplx(a), &lda, &stride_a, mklcplx(b), &ldb, &stride_b, mklcplx(&beta),
                            mklcplx(c), &ldc, &stride_c, &batch_count);
      }
#else
      for (int i = 0; i < batch_count; ++i)
        gemm(op_a, op_b, m, n, k, alpha, a + i * stride_a, lda, b + i * stride_b, ldb, beta, c + i * stride_c, ldc);
#endif
    }

  } // namespace

  void axpy(int n, double alpha, const double *x, int incx, double *y, int incy) { F77_daxpy(&n, &alpha, x, &incx, y, &incy); }
  void axpy(int n, std::complex<double> alpha, const std::complex<double> *x, int incx, std::complex<double> *y, int incy) {
    F77_zaxpy(&n, blacplx(&alpha), blacplx(x), &incx, blacplx(y), &incy);
  }

  // No Const In Wrapping!
  void copy(int n, const double *x, int incx, double *y, int incy) { F77_dcopy(&n, x, &incx, y, &incy); }
  void copy(int n, const std::complex<double> *x, int incx, std::complex<double> *y, int incy) {
    F77_zcopy(&n, blacplx(x), &incx, blacplx(y), &incy);
  }

  // dot and dotc
  float dot(int m, const float *x, int incx, const float *y, int incy) { return F77_sdot(&m, x, &incx, y, &incy); }
  std::complex<float> dot(int m, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy) {
#ifdef NDA_USE_MKL
    MKL_Complex8 result;
    cblas_cdotu_sub(m, mklcplx(x), incx, mklcplx(y), incy, &result);
#else
    auto result = F77_cdotu(&m, blacplx(x), &incx, blacplx(y), &incy);
#endif
    return std::complex<float>{result.real, result.imag};
  }
  std::complex<float> dotc(int m, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy) {
#ifdef NDA_USE_MKL
    MKL_Complex8 result;
    cblas_cdotc_sub(m, mklcplx(x), incx, mklcplx(y), incy, &result);
#else
    auto result = F77_cdotc(&m, blacplx(x), &incx, blacplx(y), &incy);
#endif
    return std::complex<float>{result.real, result.imag};
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

  // gemm
  void gemm(char op_a, char op_b, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    F77_sgemm(&op_a, &op_b, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  }
  void gemm(char op_a, char op_b, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *b,
            int ldb, std::complex<float> beta, std::complex<float> *c, int ldc) {
    F77_cgemm(&op_a, &op_b, &m, &n, &k, blacplx(&alpha), blacplx(a), &lda, blacplx(b), &ldb, blacplx(&beta), blacplx(c), &ldc);
  }
  void gemm(char op_a, char op_b, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta, double *c,
            int ldc) {
    F77_dgemm(&op_a, &op_b, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  }
  void gemm(char op_a, char op_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda,
            const std::complex<double> *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc) {
    F77_zgemm(&op_a, &op_b, &m, &n, &k, blacplx(&alpha), blacplx(a), &lda, blacplx(b), &ldb, blacplx(&beta), blacplx(c), &ldc);
  }

  // gemm_batch
  void gemm_batch(char op_a, char op_b, int m, int n, int k, float alpha, const float **a, int lda, const float **b, int ldb, float beta, float **c,
                  int ldc, int batch_count) {
    gemm_batch_impl(op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count);
  }
  void gemm_batch(char op_a, char op_b, int m, int n, int k, std::complex<float> alpha, const std::complex<float> **a, int lda,
                  const std::complex<float> **b, int ldb, std::complex<float> beta, std::complex<float> **c, int ldc, int batch_count) {
    gemm_batch_impl(op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count);
  }
  void gemm_batch(char op_a, char op_b, int m, int n, int k, double alpha, const double **a, int lda, const double **b, int ldb, double beta,
                  double **c, int ldc, int batch_count) {
    gemm_batch_impl(op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count);
  }
  void gemm_batch(char op_a, char op_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double> **a, int lda,
                  const std::complex<double> **b, int ldb, std::complex<double> beta, std::complex<double> **c, int ldc, int batch_count) {
    gemm_batch_impl(op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count);
  }

  // gemm_vbatch
  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, float alpha, const float **a, int *lda, const float **b, int *ldb, float beta,
                   float **c, int *ldc, int batch_count) {
    gemm_vbatch_impl(op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count);
  }
  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, std::complex<float> alpha, const std::complex<float> **a, int *lda,
                   const std::complex<float> **b, int *ldb, std::complex<float> beta, std::complex<float> **c, int *ldc, int batch_count) {
    gemm_vbatch_impl(op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count);
  }
  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, double alpha, const double **a, int *lda, const double **b, int *ldb, double beta,
                   double **c, int *ldc, int batch_count) {
    gemm_vbatch_impl(op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count);
  }
  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, std::complex<double> alpha, const std::complex<double> **a, int *lda,
                   const std::complex<double> **b, int *ldb, std::complex<double> beta, std::complex<double> **c, int *ldc, int batch_count) {
    gemm_vbatch_impl(op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count);
  }

  // gemm_batch_strided
  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, float alpha, const float *a, int lda, int stride_a, const float *b, int ldb,
                          int stride_b, float beta, float *c, int ldc, int stride_c, int batch_count) {
    gemm_batch_strided_impl(op_a, op_b, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_count);
  }
  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda, int stride_a,
                          const std::complex<float> *b, int ldb, int stride_b, std::complex<float> beta, std::complex<float> *c, int ldc,
                          int stride_c, int batch_count) {
    gemm_batch_strided_impl(op_a, op_b, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_count);
  }
  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, double alpha, const double *a, int lda, int stride_a, const double *b, int ldb,
                          int stride_b, double beta, double *c, int ldc, int stride_c, int batch_count) {
    gemm_batch_strided_impl(op_a, op_b, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_count);
  }
  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda, int stride_a,
                          const std::complex<double> *b, int ldb, int stride_b, std::complex<double> beta, std::complex<double> *c, int ldc,
                          int stride_c, int batch_count) {
    gemm_batch_strided_impl(op_a, op_b, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_count);
  }

  // gemv
  void gemv(char op, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y, int incy) {
    F77_sgemv(&op, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
  void gemv(char op, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *x, int incx,
            std::complex<float> beta, std::complex<float> *y, int incy) {
    F77_cgemv(&op, &m, &n, blacplx(&alpha), blacplx(a), &lda, blacplx(x), &incx, blacplx(&beta), blacplx(y), &incy);
  }
  void gemv(char op, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy) {
    F77_dgemv(&op, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
  void gemv(char op, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda, const std::complex<double> *x, int incx,
            std::complex<double> beta, std::complex<double> *y, int incy) {
    F77_zgemv(&op, &m, &n, blacplx(&alpha), blacplx(a), &lda, blacplx(x), &incx, blacplx(&beta), blacplx(y), &incy);
  }

  // ger and gerc
  void ger(int m, int n, float alpha, const float *x, int incx, const float *y, int incy, float *a, int lda) {
    F77_sger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  }
  void ger(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy,
           std::complex<float> *a, int lda) {
    F77_cgeru(&m, &n, blacplx(&alpha), blacplx(x), &incx, blacplx(y), &incy, blacplx(a), &lda);
  }
  void gerc(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy,
            std::complex<float> *a, int lda) {
    F77_cgerc(&m, &n, blacplx(&alpha), blacplx(x), &incx, blacplx(y), &incy, blacplx(a), &lda);
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

  // scal
  void scal(int m, float alpha, float *x, int incx) { F77_sscal(&m, &alpha, x, &incx); }
  void scal(int m, std::complex<float> alpha, std::complex<float> *x, int incx) { F77_cscal(&m, blacplx(&alpha), blacplx(x), &incx); }
  void scal(int m, double alpha, double *x, int incx) { F77_dscal(&m, &alpha, x, &incx); }
  void scal(int m, std::complex<double> alpha, std::complex<double> *x, int incx) { F77_zscal(&m, blacplx(&alpha), blacplx(x), &incx); }

  void swap(int n, double *x, int incx, double *y, int incy) { F77_dswap(&n, x, &incx, y, &incy); } // NOLINT (this is a BLAS swap)
  void swap(int n, std::complex<double> *x, int incx, std::complex<double> *y, int incy) {          // NOLINT (this is a BLAS swap)
    F77_zswap(&n, blacplx(x), &incx, blacplx(y), &incy);
  }

} // namespace nda::blas::f77
