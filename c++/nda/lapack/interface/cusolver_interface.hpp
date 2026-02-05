// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a C++ interface for the GPU versions of various LAPACK routines.
 */

#pragma once

#ifndef NDA_HAVE_CUDA
#error "CUDA support is not enabled in this build of nda. Please configure and install nda with -DCUDASupport=ON"
#endif

#include "../../blas/tools.hpp"

namespace nda::lapack::device {

  void gesvd(char jobu, char jobvt, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt, double *work, int lwork,
             double *rwork, int &info);
  void gesvd(char jobu, char jobvt, int m, int n, std::complex<double> *a, int lda, double *s, std::complex<double> *u, int ldu,
             std::complex<double> *vt, int ldvt, std::complex<double> *work, int lwork, double *rwork, int &info);

  void getrf(int m, int n, double *a, int lda, int *ipiv, int &info);
  void getrf(int m, int n, std::complex<double> *a, int lda, int *ipiv, int &info);

  void getrs(char op, int n, int nrhs, double const *a, int lda, int const *ipiv, double *b, int ldb, int &info);
  void getrs(char op, int n, int nrhs, std::complex<double> const *a, int lda, int const *ipiv, std::complex<double> *b, int ldb, int &info);

} // namespace nda::lapack::device
