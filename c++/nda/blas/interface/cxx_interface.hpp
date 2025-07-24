// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a C++ interface for various BLAS routines.
 */

#pragma once

#include "../tools.hpp"

#if defined(NDA_HAVE_CUDA)
#include "./cublas_interface.hpp"
#endif // NDA_HAVE_CUDA

namespace nda::blas::f77 {

  void axpy(int N, float alpha, const float *x, int incx, float *Y, int incy);
  void axpy(int N, scomplex alpha, const scomplex *x, int incx, scomplex *Y, int incy);
  void axpy(int N, double alpha, const double *x, int incx, double *Y, int incy);
  void axpy(int N, dcomplex alpha, const dcomplex *x, int incx, dcomplex *Y, int incy);

  void copy(int N, const float *x, int incx, float *Y, int incy);
  void copy(int N, const scomplex *x, int incx, scomplex *Y, int incy);
  void copy(int N, const double *x, int incx, double *Y, int incy);
  void copy(int N, const dcomplex *x, int incx, dcomplex *Y, int incy);

  float dot(int M, const float *x, int incx, const float *Y, int incy);
  scomplex dot(int M, const scomplex *x, int incx, const scomplex *Y, int incy);
  scomplex dotc(int M, const scomplex *x, int incx, const scomplex *Y, int incy);
  double dot(int M, const double *x, int incx, const double *Y, int incy);
  dcomplex dot(int M, const dcomplex *x, int incx, const dcomplex *Y, int incy);
  dcomplex dotc(int M, const dcomplex *x, int incx, const dcomplex *Y, int incy);

  void gemm(char op_a, char op_b, int M, int N, int K, float alpha, const float *A, int LDA, const float *B, int LDB, float beta, float *C, int LDC);
  void gemm(char op_a, char op_b, int M, int N, int K, scomplex alpha, const scomplex *A, int LDA, const scomplex *B, int LDB, scomplex beta,
            scomplex *C, int LDC);
  void gemm(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, const double *B, int LDB, double beta, double *C,
            int LDC);
  void gemm(char op_a, char op_b, int M, int N, int K, dcomplex alpha, const dcomplex *A, int LDA, const dcomplex *B, int LDB, dcomplex beta,
            dcomplex *C, int LDC);

  void gemm_batch(char op_a, char op_b, int M, int N, int K, float alpha, const float **A, int LDA, const float **B, int LDB, float beta, float **C,
                  int LDC, int batch_count);
  void gemm_batch(char op_a, char op_b, int M, int N, int K, scomplex alpha, const scomplex **A, int LDA, const scomplex **B, int LDB, scomplex beta,
                  scomplex **C, int LDC, int batch_count);
  void gemm_batch(char op_a, char op_b, int M, int N, int K, double alpha, const double **A, int LDA, const double **B, int LDB, double beta,
                  double **C, int LDC, int batch_count);
  void gemm_batch(char op_a, char op_b, int M, int N, int K, dcomplex alpha, const dcomplex **A, int LDA, const dcomplex **B, int LDB, dcomplex beta,
                  dcomplex **C, int LDC, int batch_count);

  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, float alpha, const float **A, int *LDA, const float **B, int *LDB, float beta,
                   float **C, int *LDC, int batch_count);
  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, scomplex alpha, const scomplex **A, int *LDA, const scomplex **B, int *LDB,
                   scomplex beta, scomplex **C, int *LDC, int batch_count);
  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, double alpha, const double **A, int *LDA, const double **B, int *LDB, double beta,
                   double **C, int *LDC, int batch_count);
  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, dcomplex alpha, const dcomplex **A, int *LDA, const dcomplex **B, int *LDB,
                   dcomplex beta, dcomplex **C, int *LDC, int batch_count);

  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, float alpha, const float *A, int LDA, int strideA, const float *B, int LDB,
                          int strideB, float beta, float *C, int LDC, int strideC, int batch_count);
  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, scomplex alpha, const scomplex *A, int LDA, int strideA, const scomplex *B,
                          int LDB, int srideB, scomplex beta, scomplex *C, int LDC, int strideC, int batch_count);
  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, int strideA, const double *B, int LDB,
                          int strideB, double beta, double *C, int LDC, int strideC, int batch_count);
  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, dcomplex alpha, const dcomplex *A, int LDA, int strideA, const dcomplex *B,
                          int LDB, int srideB, dcomplex beta, dcomplex *C, int LDC, int strideC, int batch_count);

  void gemv(char op, int M, int N, float alpha, const float *A, int LDA, const float *x, int incx, float beta, float *Y, int incy);
  void gemv(char op, int M, int N, scomplex alpha, const scomplex *A, int LDA, const scomplex *x, int incx, scomplex beta, scomplex *Y, int incy);
  void gemv(char op, int M, int N, double alpha, const double *A, int LDA, const double *x, int incx, double beta, double *Y, int incy);
  void gemv(char op, int M, int N, dcomplex alpha, const dcomplex *A, int LDA, const dcomplex *x, int incx, dcomplex beta, dcomplex *Y, int incy);

  void ger(int M, int N, float alpha, const float *x, int incx, const float *Y, int incy, float *A, int LDA);
  void ger(int M, int N, scomplex alpha, const scomplex *x, int incx, const scomplex *Y, int incy, scomplex *A, int LDA);
  void gerc(int M, int N, scomplex alpha, const scomplex *x, int incx, const scomplex *Y, int incy, scomplex *A, int LDA);
  void ger(int M, int N, double alpha, const double *x, int incx, const double *Y, int incy, double *A, int LDA);
  void ger(int M, int N, dcomplex alpha, const dcomplex *x, int incx, const dcomplex *Y, int incy, dcomplex *A, int LDA);
  void gerc(int M, int N, dcomplex alpha, const dcomplex *x, int incx, const dcomplex *Y, int incy, dcomplex *A, int LDA);

  void scal(int M, float alpha, float *x, int incx);
  void scal(int M, scomplex alpha, scomplex *x, int incx);
  void scal(int M, double alpha, double *x, int incx);
  void scal(int M, dcomplex alpha, dcomplex *x, int incx);

  void swap(int N, float *x, int incx, float *Y, int incy);       // NOLINT (this is a BLAS swap)
  void swap(int N, scomplex *x, int incx, scomplex *Y, int incy); // NOLINT (this is a BLAS swap)
  void swap(int N, double *x, int incx, double *Y, int incy);     // NOLINT (this is a BLAS swap)
  void swap(int N, dcomplex *x, int incx, dcomplex *Y, int incy); // NOLINT (this is a BLAS swap)

} // namespace nda::blas::f77
