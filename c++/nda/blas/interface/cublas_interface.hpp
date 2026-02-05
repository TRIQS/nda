// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a C++ interface for the GPU versions of various BLAS routines.
 */

#pragma once

#include "../tools.hpp"

#ifndef NDA_HAVE_CUDA
#error "CUDA support is not enabled in this build of nda. Please configure and install nda with -DCUDASupport=ON"
#endif

#ifndef NDA_HAVE_MAGMA
#include "../../exceptions.hpp"
#endif // NDA_HAVE_MAGMA

namespace nda::blas::device {

  void axpy(int n, double alpha, const double *x, int incx, double *y, int incy);
  void axpy(int n, dcomplex alpha, const dcomplex *x, int incx, dcomplex *y, int incy);

  void copy(int n, const double *x, int incx, double *y, int incy);
  void copy(int n, const dcomplex *x, int incx, dcomplex *y, int incy);

  double dot(int m, const double *x, int incx, const double *y, int incy);
  dcomplex dot(int m, const dcomplex *x, int incx, const dcomplex *y, int incy);
  dcomplex dotc(int m, const dcomplex *x, int incx, const dcomplex *y, int incy);

  void gemm(char op_a, char op_b, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta, double *c,
            int ldc);
  void gemm(char op_a, char op_b, int m, int n, int k, dcomplex alpha, const dcomplex *a, int lda, const dcomplex *b, int ldb, dcomplex beta,
            dcomplex *c, int ldc);

  void gemm_batch(char op_a, char op_b, int m, int n, int k, double alpha, const double **a, int lda, const double **b, int ldb, double beta,
                  double **c, int ldc, int batch_count);
  void gemm_batch(char op_a, char op_b, int m, int n, int k, dcomplex alpha, const dcomplex **a, int lda, const dcomplex **b, int ldb, dcomplex beta,
                  dcomplex **c, int ldc, int batch_count);

#ifdef NDA_HAVE_MAGMA
  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, double alpha, const double **a, int *lda, const double **b, int *ldb, double beta,
                   double **c, int *ldc, int batch_count);
  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, dcomplex alpha, const dcomplex **a, int *lda, const dcomplex **b, int *ldb,
                   dcomplex beta, dcomplex **c, int *ldc, int batch_count);
#else
  inline void gemm_vbatch(char, char, int *, int *, int *, double, const double **, int *, const double **, int *, double, double **, int *, int) {
    NDA_RUNTIME_ERROR << "nda::blas::device::gemmv_batch requires Magma [https://icl.cs.utk.edu/magma/]. Configure nda with -DMagmaSupport=ON";
  }
  inline void gemm_vbatch(char, char, int *, int *, int *, dcomplex, const dcomplex **, int *, const dcomplex **, int *, dcomplex, dcomplex **, int *,
                          int) {
    NDA_RUNTIME_ERROR << "nda::blas::device::gemmv_batch requires Magma [https://icl.cs.utk.edu/magma/]. Configure nda with -DMagmaSupport=ON";
  }
#endif

  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, double alpha, const double *a, int lda, int stride_a, const double *b, int ldb,
                          int stride_b, double beta, double *c, int ldc, int stride_c, int batch_count);
  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, dcomplex alpha, const dcomplex *a, int lda, int stride_a, const dcomplex *b,
                          int ldb, int stride_b, dcomplex beta, dcomplex *c, int ldc, int stride_c, int batch_count);

  void gemv(char op, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy);
  void gemv(char op, int m, int n, dcomplex alpha, const dcomplex *a, int lda, const dcomplex *x, int incx, dcomplex beta, dcomplex *y, int incy);

  void ger(int m, int n, double alpha, const double *x, int incx, const double *y, int incy, double *a, int lda);
  void ger(int m, int n, dcomplex alpha, const dcomplex *x, int incx, const dcomplex *y, int incy, dcomplex *a, int lda);
  void gerc(int m, int n, dcomplex alpha, const dcomplex *x, int incx, const dcomplex *y, int incy, dcomplex *a, int lda);

  void scal(int m, double alpha, double *x, int incx);
  void scal(int m, dcomplex alpha, dcomplex *x, int incx);

  void swap(int n, double *x, int incx, double *y, int incy);     // NOLINT (this is a BLAS swap)
  void swap(int n, dcomplex *x, int incx, dcomplex *y, int incy); // NOLINT (this is a BLAS swap)

} // namespace nda::blas::device
