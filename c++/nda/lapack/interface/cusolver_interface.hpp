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

  void gesvd(char JOBU, char JOBVT, int M, int N, float *A, int LDA, float *S, float *U, int LDU, float *VT, int LDVT, float *WORK, int LWORK,
             float *RWORK, int &INFO);
  void gesvd(char JOBU, char JOBVT, int M, int N, fcomplex *A, int LDA, float *S, fcomplex *U, int LDU, fcomplex *VT, int LDVT, fcomplex *WORK,
             int LWORK, float *RWORK, int &INFO);
  void gesvd(char JOBU, char JOBVT, int M, int N, double *A, int LDA, double *S, double *U, int LDU, double *VT, int LDVT, double *WORK, int LWORK,
             double *RWORK, int &INFO);
  void gesvd(char JOBU, char JOBVT, int M, int N, dcomplex *A, int LDA, double *S, dcomplex *U, int LDU, dcomplex *VT, int LDVT, dcomplex *WORK,
             int LWORK, double *RWORK, int &INFO);

  int getrf_bufferSize(int M, int N, float *A, int LDA);
  int getrf_bufferSize(int M, int N, double *A, int LDA);
  int getrf_bufferSize(int M, int N, std::complex<float> *A, int LDA);
  int getrf_bufferSize(int M, int N, dcomplex *A, int LDA);

  void getrf(int M, int N, float *A, int LDA, float *W, int *ipiv, int &info);
  void getrf(int M, int N, double *A, int LDA, double *W, int *ipiv, int &info);
  void getrf(int M, int N, std::complex<float> *A, int LDA, std::complex<float> *W, int *ipiv, int &info);
  void getrf(int M, int N, dcomplex *A, int LDA, dcomplex *W, int *ipiv, int &info);

  template <typename T>
  int getri_bufferSize(int N, T *, int) {
    return N * N;
  };

  void getri(int N, float *A, int LDA, int const *ipiv, float *WORK, int LWORK, int &info);
  void getri(int N, double *A, int LDA, int const *ipiv, double *WORK, int LWORK, int &info);
  void getri(int N, std::complex<float> *A, int LDA, int const *ipiv, std::complex<float> *WORK, int LWORK, int &info);
  void getri(int N, dcomplex *A, int LDA, int const *ipiv, dcomplex *WORK, int LWORK, int &info);

  void getrs(char op, int N, int NRHS, float const *A, int LDA, int const *ipiv, float *B, int LDB, int &info);
  void getrs(char op, int N, int NRHS, double const *A, int LDA, int const *ipiv, double *B, int LDB, int &info);
  void getrs(char op, int N, int NRHS, std::complex<float> const *A, int LDA, int const *ipiv, std::complex<float> *B, int LDB, int &info);
  void getrs(char op, int N, int NRHS, dcomplex const *A, int LDA, int const *ipiv, dcomplex *B, int LDB, int &info);

  int geqrf_bufferSize(int M, int N, float *A, int LDA);
  int geqrf_bufferSize(int M, int N, double *A, int LDA);
  int geqrf_bufferSize(int M, int N, std::complex<float> *A, int LDA);
  int geqrf_bufferSize(int M, int N, dcomplex *A, int LDA);

  void geqrf(int M, int N, float *A, int LDA, float *tau, float *W, int Lwork, int &info);
  void geqrf(int M, int N, double *A, int LDA, double *tau, double *W, int Lwork, int &info);
  void geqrf(int M, int N, std::complex<float> *A, int LDA, std::complex<float> *tau, std::complex<float> *W, int Lwork, int &info);
  void geqrf(int M, int N, dcomplex *A, int LDA, dcomplex *tau, dcomplex *W, int Lwork, int &info);

  int orgqr_bufferSize(int M, int N, int K, const float *A, int LDA, const float *tau);
  int orgqr_bufferSize(int M, int N, int K, const double *A, int LDA, const double *tau);
  int ungqr_bufferSize(int M, int N, int K, const std::complex<float> *A, int LDA, const std::complex<float> *tau);
  int ungqr_bufferSize(int M, int N, int K, const dcomplex *A, int LDA, const dcomplex *tau);

  void orgqr(int M, int N, int K, float *A, int LDA, const float *tau, float *work, int Lwork, int &info);
  void orgqr(int M, int N, int K, double *A, int LDA, const double *tau, double *work, int Lwork, int &info);
  void ungqr(int M, int N, int K, std::complex<float> *A, int LDA, const std::complex<float> *tau, std::complex<float> *work, int Lwork, int &info);
  void ungqr(int M, int N, int K, dcomplex *A, int LDA, const dcomplex *tau, dcomplex *work, int Lwork, int &info);

} // namespace nda::lapack::device
