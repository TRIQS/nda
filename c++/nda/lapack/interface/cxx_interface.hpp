// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a C++ interface for various LAPACK routines.
 */

#pragma once

#include "../../blas/tools.hpp"

#include <complex>

#if defined(NDA_HAVE_CUDA)
#include "./cusolver_interface.hpp"
#endif

namespace nda::lapack::f77 {

  void gelss(int M, int N, int NRHS, float *A, int LDA, float *B, int LDB, float *S, float RCOND, int &RANK, float *WORK, int LWORK, float *RWORK,
             int &INFO);
  void gelss(int M, int N, int NRHS, std::complex<float> *A, int LDA, std::complex<float> *B, int LDB, float *S, float RCOND, int &RANK,
             std::complex<float> *WORK, int LWORK, float *RWORK, int &INFO);
  void gelss(int M, int N, int NRHS, double *A, int LDA, double *B, int LDB, double *S, double RCOND, int &RANK, double *WORK, int LWORK,
             double *RWORK, int &INFO);
  void gelss(int M, int N, int NRHS, std::complex<double> *A, int LDA, std::complex<double> *B, int LDB, double *S, double RCOND, int &RANK,
             std::complex<double> *WORK, int LWORK, double *RWORK, int &INFO);

  void gesvd(char JOBU, char JOBVT, int M, int N, float *A, int LDA, float *S, float *U, int LDU, float *VT, int LDVT, float *WORK, int LWORK,
             float *RWORK, int &INFO);
  void gesvd(char JOBU, char JOBVT, int M, int N, std::complex<float> *A, int LDA, float *S, std::complex<float> *U, int LDU, std::complex<float> *VT,
             int LDVT, std::complex<float> *WORK, int LWORK, float *RWORK, int &INFO);
  void gesvd(char JOBU, char JOBVT, int M, int N, double *A, int LDA, double *S, double *U, int LDU, double *VT, int LDVT, double *WORK, int LWORK,
             double *RWORK, int &INFO);
  void gesvd(char JOBU, char JOBVT, int M, int N, std::complex<double> *A, int LDA, double *S, std::complex<double> *U, int LDU,
             std::complex<double> *VT, int LDVT, std::complex<double> *WORK, int LWORK, double *RWORK, int &INFO);

  void geqp3(int M, int N, float *A, int LDA, int *JPVT, float *TAU, float *WORK, int LWORK, float *RWORK, int &INFO);
  void geqp3(int M, int N, std::complex<float> *A, int LDA, int *JPVT, std::complex<float> *TAU, std::complex<float> *WORK, int LWORK, float *RWORK,
             int &INFO);
  void geqp3(int M, int N, double *A, int LDA, int *JPVT, double *TAU, double *WORK, int LWORK, double *RWORK, int &INFO);
  void geqp3(int M, int N, std::complex<double> *A, int LDA, int *JPVT, std::complex<double> *TAU, std::complex<double> *WORK, int LWORK,
             double *RWORK, int &INFO);

  void orgqr(int M, int N, int K, float *A, int LDA, float *TAU, float *WORK, int LWORK, int &INFO);
  void orgqr(int M, int N, int K, double *A, int LDA, double *TAU, double *WORK, int LWORK, int &INFO);

  void ungqr(int M, int N, int K, std::complex<float> *A, int LDA, std::complex<float> *TAU, std::complex<float> *WORK, int LWORK, int &INFO);
  void ungqr(int M, int N, int K, std::complex<double> *A, int LDA, std::complex<double> *TAU, std::complex<double> *WORK, int LWORK, int &INFO);

  void getrf(int M, int N, float *A, int LDA, int *ipiv, int &info);
  void getrf(int M, int N, std::complex<float> *A, int LDA, int *ipiv, int &info);
  void getrf(int M, int N, double *A, int LDA, int *ipiv, int &info);
  void getrf(int M, int N, std::complex<double> *A, int LDA, int *ipiv, int &info);

  void getri(int N, float *A, int LDA, int const *ipiv, float *work, int lwork, int &info);
  void getri(int N, std::complex<float> *A, int LDA, int const *ipiv, std::complex<float> *work, int lwork, int &info);
  void getri(int N, double *A, int LDA, int const *ipiv, double *work, int lwork, int &info);
  void getri(int N, std::complex<double> *A, int LDA, int const *ipiv, std::complex<double> *work, int lwork, int &info);

  void gtsv(int N, int NRHS, double *DL, double *D, double *DU, double *B, int LDB, int &info);
  void gtsv(int N, int NRHS, std::complex<double> *DL, std::complex<double> *D, std::complex<double> *DU, std::complex<double> *B, int LDB,
            int &info);

  void stev(char J, int N, double *D, double *E, double *Z, int ldz, double *work, int &info);

  void syev(char JOBZ, char UPLO, int N, float *A, int LDA, float *W, float *work, int lwork, int &info);
  void syev(char JOBZ, char UPLO, int N, double *A, int LDA, double *W, double *work, int lwork, int &info);

  void heev(char JOBZ, char UPLO, int N, std::complex<float> *A, int LDA, float *W, std::complex<float> *work, int lwork, float *rwork, int &info);
  void heev(char JOBZ, char UPLO, int N, std::complex<double> *A, int LDA, double *W, std::complex<double> *work, int lwork, double *rwork,
            int &info);

  void sygv(int ITYPE, char JOBZ, char UPLO, int N, float *A, int LDA, float *B, int LDB, float *W, float *work, int lwork, int &info);
  void sygv(int ITYPE, char JOBZ, char UPLO, int N, double *A, int LDA, double *B, int LDB, double *W, double *work, int lwork, int &info);

  void hegv(int ITYPE, char JOBZ, char UPLO, int N, std::complex<float> *A, int LDA, std::complex<float> *B, int LDB, float *W,
            std::complex<float> *work, int lwork, float *rwork, int &info);
  void hegv(int ITYPE, char JOBZ, char UPLO, int N, std::complex<double> *A, int LDA, std::complex<double> *B, int LDB, double *W,
            std::complex<double> *work, int lwork, double *rwork, int &info);

  void getrs(char op, int N, int NRHS, float const *A, int LDA, int const *ipiv, float *B, int LDB, int &info);
  void getrs(char op, int N, int NRHS, std::complex<float> const *A, int LDA, int const *ipiv, std::complex<float> *B, int LDB, int &info);
  void getrs(char op, int N, int NRHS, double const *A, int LDA, int const *ipiv, double *B, int LDB, int &info);
  void getrs(char op, int N, int NRHS, std::complex<double> const *A, int LDA, int const *ipiv, std::complex<double> *B, int LDB, int &info);

} // namespace nda::lapack::f77

// Useful routines from the BLAS interface
namespace nda::lapack {

  // See nda::blas::get_ld.
  using blas::get_ld;

  // See nda::blas::get_ncols.
  using blas::get_ncols;

  // See nda::blas::get_op.
  using blas::get_op;

  // See nda::blas::has_C_layout.
  using blas::has_C_layout;

  // See nda::blas::has_F_layout.
  using blas::has_F_layout;

  // See nda::blas::is_conj_array_expr.
  using blas::is_conj_array_expr;

  // See nda::blas::get_array.
  using blas::get_array;

} // namespace nda::lapack
