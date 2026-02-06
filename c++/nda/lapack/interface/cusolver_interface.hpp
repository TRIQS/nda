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

  void getrf(int m, int n, double *a, int lda, int *ipiv, int &info);
  void getrf(int m, int n, std::complex<double> *a, int lda, int *ipiv, int &info);

  void getrs(char op, int n, int nrhs, double const *a, int lda, int const *ipiv, double *b, int ldb, int &info);
  void getrs(char op, int n, int nrhs, std::complex<double> const *a, int lda, int const *ipiv, std::complex<double> *b, int ldb, int &info);

} // namespace nda::lapack::device
