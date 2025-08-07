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

  void axpy(int N, double alpha, const double *x, int incx, double *Y, int incy);
  void axpy(int N, dcomplex alpha, const dcomplex *x, int incx, dcomplex *Y, int incy);

  void copy(int N, const double *x, int incx, double *Y, int incy);
  void copy(int N, const dcomplex *x, int incx, dcomplex *Y, int incy);

  double dot(int M, const double *x, int incx, const double *Y, int incy);
  dcomplex dot(int M, const dcomplex *x, int incx, const dcomplex *Y, int incy);
  dcomplex dotc(int M, const dcomplex *x, int incx, const dcomplex *Y, int incy);

  void gemm(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, const double *B, int LDB, double beta, double *C,
            int LDC);
  void gemm(char op_a, char op_b, int M, int N, int K, dcomplex alpha, const dcomplex *A, int LDA, const dcomplex *B, int LDB, dcomplex beta,
            dcomplex *C, int LDC);

  void gemm_batch(char op_a, char op_b, int M, int N, int K, double alpha, const double **A, int LDA, const double **B, int LDB, double beta,
                  double **C, int LDC, int batch_count);
  void gemm_batch(char op_a, char op_b, int M, int N, int K, dcomplex alpha, const dcomplex **A, int LDA, const dcomplex **B, int LDB, dcomplex beta,
                  dcomplex **C, int LDC, int batch_count);

#ifdef NDA_HAVE_MAGMA
  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, double alpha, const double **A, int *LDA, const double **B, int *LDB, double beta,
                   double **C, int *LDC, int batch_count);
  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, dcomplex alpha, const dcomplex **A, int *LDA, const dcomplex **B, int *LDB,
                   dcomplex beta, dcomplex **C, int *LDC, int batch_count);
#else
  inline void gemm_vbatch(char, char, int *, int *, int *, double, const double **, int *, const double **, int *, double, double **, int *, int) {
    NDA_RUNTIME_ERROR << "nda::blas::device::gemmv_batch requires Magma [https://icl.cs.utk.edu/magma/]. Configure nda with -DUse_Magma=ON";
  }
  inline void gemm_vbatch(char, char, int *, int *, int *, dcomplex, const dcomplex **, int *, const dcomplex **, int *, dcomplex, dcomplex **, int *,
                          int) {
    NDA_RUNTIME_ERROR << "nda::blas::device::gemmv_batch requires Magma [https://icl.cs.utk.edu/magma/]. Configure nda with -DUse_Magma=ON";
  }
#endif

  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, int strideA, const double *B, int LDB,
                          int strideB, double beta, double *C, int LDC, int strideC, int batch_count);
  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, dcomplex alpha, const dcomplex *A, int LDA, int strideA, const dcomplex *B,
                          int LDB, int srideB, dcomplex beta, dcomplex *C, int LDC, int strideC, int batch_count);

  void gemv(char op, int M, int N, double alpha, const double *A, int LDA, const double *x, int incx, double beta, double *Y, int incy);
  void gemv(char op, int M, int N, dcomplex alpha, const dcomplex *A, int LDA, const dcomplex *x, int incx, dcomplex beta, dcomplex *Y, int incy);

  void ger(int M, int N, double alpha, const double *x, int incx, const double *Y, int incy, double *A, int LDA);
  void ger(int M, int N, dcomplex alpha, const dcomplex *x, int incx, const dcomplex *Y, int incy, dcomplex *A, int LDA);
  void gerc(int M, int N, dcomplex alpha, const dcomplex *x, int incx, const dcomplex *Y, int incy, dcomplex *A, int LDA);

  void scal(int M, double alpha, double *x, int incx);
  void scal(int M, dcomplex alpha, dcomplex *x, int incx);

  void swap(int N, double *x, int incx, double *Y, int incy);     // NOLINT (this is a BLAS swap)
  void swap(int N, dcomplex *x, int incx, dcomplex *Y, int incy); // NOLINT (this is a BLAS swap)

} // namespace nda::blas::device
