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

namespace nda::blas::device {

  void set_synchronization(bool do_sync) noexcept;
  bool get_synchronization() noexcept;

  void axpy(int n, float alpha, const float *x, int incx, float *y, int incy);
  void axpy(int n, std::complex<float> alpha, const std::complex<float> *x, int incx, std::complex<float> *y, int incy);
  void axpy(int n, double alpha, const double *x, int incx, double *y, int incy);
  void axpy(int n, std::complex<double> alpha, const std::complex<double> *x, int incx, std::complex<double> *y, int incy);

  void copy(int n, const float *x, int incx, float *y, int incy);
  void copy(int n, const std::complex<float> *x, int incx, std::complex<float> *y, int incy);
  void copy(int n, const double *x, int incx, double *y, int incy);
  void copy(int n, const std::complex<double> *x, int incx, std::complex<double> *y, int incy);

  float dot(int m, const float *x, int incx, const float *y, int incy);
  std::complex<float> dot(int m, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy);
  std::complex<float> dotc(int m, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy);
  double dot(int m, const double *x, int incx, const double *y, int incy);
  std::complex<double> dot(int m, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy);
  std::complex<double> dotc(int m, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy);

  void gemm(char op_a, char op_b, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc);
  void gemm(char op_a, char op_b, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *b,
            int ldb, std::complex<float> beta, std::complex<float> *c, int ldc);
  void gemm(char op_a, char op_b, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta, double *c,
            int ldc);
  void gemm(char op_a, char op_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda,
            const std::complex<double> *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc);

  void gemm_batch(char op_a, char op_b, int m, int n, int k, float alpha, const float **a, int lda, const float **b, int ldb, float beta, float **c,
                  int ldc, int batch_count);
  void gemm_batch(char op_a, char op_b, int m, int n, int k, std::complex<float> alpha, const std::complex<float> **a, int lda,
                  const std::complex<float> **b, int ldb, std::complex<float> beta, std::complex<float> **c, int ldc, int batch_count);
  void gemm_batch(char op_a, char op_b, int m, int n, int k, double alpha, const double **a, int lda, const double **b, int ldb, double beta,
                  double **c, int ldc, int batch_count);
  void gemm_batch(char op_a, char op_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double> **a, int lda,
                  const std::complex<double> **b, int ldb, std::complex<double> beta, std::complex<double> **c, int ldc, int batch_count);

  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, float alpha, const float **a, int *lda, const float **b, int *ldb, float beta,
                   float **c, int *ldc, int batch_count);
  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, std::complex<float> alpha, const std::complex<float> **a, int *lda,
                   const std::complex<float> **b, int *ldb, std::complex<float> beta, std::complex<float> **c, int *ldc, int batch_count);
  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, double alpha, const double **a, int *lda, const double **b, int *ldb, double beta,
                   double **c, int *ldc, int batch_count);
  void gemm_vbatch(char op_a, char op_b, int *m, int *n, int *k, std::complex<double> alpha, const std::complex<double> **a, int *lda,
                   const std::complex<double> **b, int *ldb, std::complex<double> beta, std::complex<double> **c, int *ldc, int batch_count);

  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, float alpha, const float *a, int lda, int stride_a, const float *b, int ldb,
                          int stride_b, float beta, float *c, int ldc, int stride_c, int batch_count);
  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda, int stride_a,
                          const std::complex<float> *b, int ldb, int stride_b, std::complex<float> beta, std::complex<float> *c, int ldc,
                          int stride_c, int batch_count);
  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, double alpha, const double *a, int lda, int stride_a, const double *b, int ldb,
                          int stride_b, double beta, double *c, int ldc, int stride_c, int batch_count);
  void gemm_batch_strided(char op_a, char op_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda, int stride_a,
                          const std::complex<double> *b, int ldb, int stride_b, std::complex<double> beta, std::complex<double> *c, int ldc,
                          int stride_c, int batch_count);

  void gemv(char op, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y, int incy);
  void gemv(char op, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *x, int incx,
            std::complex<float> beta, std::complex<float> *y, int incy);
  void gemv(char op, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy);
  void gemv(char op, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda, const std::complex<double> *x, int incx,
            std::complex<double> beta, std::complex<double> *y, int incy);

  void ger(int m, int n, float alpha, const float *x, int incx, const float *y, int incy, float *a, int lda);
  void ger(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy,
           std::complex<float> *a, int lda);
  void gerc(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy,
            std::complex<float> *a, int lda);
  void ger(int m, int n, double alpha, const double *x, int incx, const double *y, int incy, double *a, int lda);
  void ger(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy,
           std::complex<double> *a, int lda);
  void gerc(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy,
            std::complex<double> *a, int lda);

  void scal(int m, float alpha, float *x, int incx);
  void scal(int m, std::complex<float> alpha, std::complex<float> *x, int incx);
  void scal(int m, double alpha, double *x, int incx);
  void scal(int m, std::complex<double> alpha, std::complex<double> *x, int incx);

  void swap(int n, float *x, int incx, float *y, int incy);                               // NOLINT (this is a BLAS swap)
  void swap(int n, std::complex<float> *x, int incx, std::complex<float> *y, int incy);   // NOLINT (this is a BLAS swap)
  void swap(int n, double *x, int incx, double *y, int incy);                             // NOLINT (this is a BLAS swap)
  void swap(int n, std::complex<double> *x, int incx, std::complex<double> *y, int incy); // NOLINT (this is a BLAS swap)

  void getrf_batch(int n, float **a_array, int lda, int *ipiv_array, int *info_array, int batch_size);
  void getrf_batch(int n, std::complex<float> **a_array, int lda, int *ipiv_array, int *info_array, int batch_size);
  void getrf_batch(int n, double **a_array, int lda, int *ipiv_array, int *info_array, int batch_size);
  void getrf_batch(int n, std::complex<double> **a_array, int lda, int *ipiv_array, int *info_array, int batch_size);

  void getri_batch(int n, float **a_array, int lda, int const *ipiv_array, float **c_array, int ldc, int *info_array, int batch_size);
  void getri_batch(int n, std::complex<float> **a_array, int lda, int const *ipiv_array, std::complex<float> **c_array, int ldc, int *info_array,
                   int batch_size);
  void getri_batch(int n, double **a_array, int lda, int const *ipiv_array, double **c_array, int ldc, int *info_array, int batch_size);
  void getri_batch(int n, std::complex<double> **a_array, int lda, int const *ipiv_array, std::complex<double> **c_array, int ldc, int *info_array,
                   int batch_size);

  void getrs_batch(char op, int n, int nrhs, const float **a_array, int lda, int const *ipiv_array, float **b_array, int ldb, int &info,
                   int batch_size);
  void getrs_batch(char op, int n, int nrhs, const std::complex<float> **a_array, int lda, int const *ipiv_array, std::complex<float> **b_array,
                   int ldb, int &info, int batch_size);
  void getrs_batch(char op, int n, int nrhs, const double **a_array, int lda, int const *ipiv_array, double **b_array, int ldb, int &info,
                   int batch_size);
  void getrs_batch(char op, int n, int nrhs, const std::complex<double> **a_array, int lda, int const *ipiv_array, std::complex<double> **b_array,
                   int ldb, int &info, int batch_size);

  void geqrf_batch(int n, int m, float **a_array, int lda, float **tau_array, int &info, int batch_size);
  void geqrf_batch(int n, int m, std::complex<float> **a_array, int lda, std::complex<float> **tau_array, int &info, int batch_size);
  void geqrf_batch(int n, int m, double **a_array, int lda, double **tau_array, int &info, int batch_size);
  void geqrf_batch(int n, int m, std::complex<double> **a_array, int lda, std::complex<double> **tau_array, int &info, int batch_size);

} // namespace nda::blas::device
