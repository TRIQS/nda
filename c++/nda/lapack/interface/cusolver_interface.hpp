// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a C++ interface for the GPU versions of various LAPACK routines.
 */

#pragma once

#include "../../blas/tools.hpp"

namespace nda::lapack::device {

  int gesvd_buffer_size(int m, int n, float *);
  int gesvd_buffer_size(int m, int n, std::complex<float> *);
  int gesvd_buffer_size(int m, int n, double *);
  int gesvd_buffer_size(int m, int n, std::complex<double> *);

  void gesvd(char jobu, char jobvt, int m, int n, float *a, int lda, float *s, float *u, int ldu, float *vt, int ldvt, float *work, int lwork,
             float *rwork, int &info);
  void gesvd(char jobu, char jobvt, int m, int n, std::complex<float> *a, int lda, float *s, std::complex<float> *u, int ldu, std::complex<float> *vt,
             int ldvt, std::complex<float> *work, int lwork, float *rwork, int &info);
  void gesvd(char jobu, char jobvt, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt, double *work, int lwork,
             double *rwork, int &info);
  void gesvd(char jobu, char jobvt, int m, int n, std::complex<double> *a, int lda, double *s, std::complex<double> *u, int ldu,
             std::complex<double> *vt, int ldvt, std::complex<double> *work, int lwork, double *rwork, int &info);

  int getrf_buffer_size(int m, int n, float *a, int lda);
  int getrf_buffer_size(int m, int n, std::complex<float> *a, int lda);
  int getrf_buffer_size(int m, int n, double *a, int lda);
  int getrf_buffer_size(int m, int n, std::complex<double> *a, int lda);

  void getrf(int m, int n, float *a, int lda, float *work, int *ipiv, int &info);
  void getrf(int m, int n, std::complex<float> *a, int lda, std::complex<float> *work, int *ipiv, int &info);
  void getrf(int m, int n, double *a, int lda, double *work, int *ipiv, int &info);
  void getrf(int m, int n, std::complex<double> *a, int lda, std::complex<double> *work, int *ipiv, int &info);

  void getrs(char op, int n, int nrhs, float const *a, int lda, int const *ipiv, float *b, int ldb, int &info);
  void getrs(char op, int n, int nrhs, std::complex<float> const *a, int lda, int const *ipiv, std::complex<float> *b, int ldb, int &info);
  void getrs(char op, int n, int nrhs, double const *a, int lda, int const *ipiv, double *b, int ldb, int &info);
  void getrs(char op, int n, int nrhs, std::complex<double> const *a, int lda, int const *ipiv, std::complex<double> *b, int ldb, int &info);

  int geqrf_buffer_size(int m, int n, float *a, int lda);
  int geqrf_buffer_size(int m, int n, std::complex<float> *a, int lda);
  int geqrf_buffer_size(int m, int n, double *a, int lda);
  int geqrf_buffer_size(int m, int n, std::complex<double> *a, int lda);

  void geqrf(int m, int n, float *a, int lda, float *tau, float *work, int lwork, int &info);
  void geqrf(int m, int n, std::complex<float> *a, int lda, std::complex<float> *tau, std::complex<float> *work, int lwork, int &info);
  void geqrf(int m, int n, double *a, int lda, double *tau, double *work, int lwork, int &info);
  void geqrf(int m, int n, std::complex<double> *a, int lda, std::complex<double> *tau, std::complex<double> *work, int lwork, int &info);

  int orgqr_buffer_size(int m, int n, int k, float const *a, int lda, float const *tau);
  int orgqr_buffer_size(int m, int n, int k, double const *a, int lda, double const *tau);

  void orgqr(int m, int n, int k, float *a, int lda, float const *tau, float *work, int lwork, int &info);
  void orgqr(int m, int n, int k, double *a, int lda, double const *tau, double *work, int lwork, int &info);

  int ungqr_buffer_size(int m, int n, int k, std::complex<float> const *a, int lda, std::complex<float> const *tau);
  int ungqr_buffer_size(int m, int n, int k, std::complex<double> const *a, int lda, std::complex<double> const *tau);

  void ungqr(int m, int n, int k, std::complex<float> *a, int lda, std::complex<float> const *tau, std::complex<float> *work, int lwork, int &info);
  void ungqr(int m, int n, int k, std::complex<double> *a, int lda, std::complex<double> const *tau, std::complex<double> *work, int lwork,
             int &info);

} // namespace nda::lapack::device
