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

  void gesvd(char JOBU, char JOBVT, int M, int N, double *A, int LDA, double *S, double *U, int LDU, double *VT, int LDVT, double *WORK, int LWORK,
             double *RWORK, int &INFO);
  void gesvd(char JOBU, char JOBVT, int M, int N, dcomplex *A, int LDA, double *S, dcomplex *U, int LDU, dcomplex *VT, int LDVT, dcomplex *WORK,
             int LWORK, double *RWORK, int &INFO);

  void getrf(int M, int N, double *A, int LDA, int *ipiv, int &info);
  void getrf(int M, int N, dcomplex *A, int LDA, int *ipiv, int &info);

  void getrs(char op, int N, int NRHS, double const *A, int LDA, int const *ipiv, double *B, int LDB, int &info);
  void getrs(char op, int N, int NRHS, dcomplex const *A, int LDA, int const *ipiv, dcomplex *B, int LDB, int &info);

} // namespace nda::lapack::device
