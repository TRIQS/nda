// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Implementation details for lapack/interface/cxx_interface.hpp.
 */

// Extracted from Reference Lapack (https://github.com/Reference-LAPACK):
#include "./lapack.h"
#include "./cxx_interface.hpp"

#include <complex>

namespace nda::lapack::f77 {

  void gelss(int M, int N, int NRHS, double *A, int LDA, double *B, int LDB, double *S, double RCOND, int &RANK, double *WORK, int LWORK,
             [[maybe_unused]] double *RWORK, int &INFO) {
    LAPACK_dgelss(&M, &N, &NRHS, A, &LDA, B, &LDB, S, &RCOND, &RANK, WORK, &LWORK, &INFO);
  }
  void gelss(int M, int N, int NRHS, std::complex<double> *A, int LDA, std::complex<double> *B, int LDB, double *S, double RCOND, int &RANK,
             std::complex<double> *WORK, int LWORK, double *RWORK, int &INFO) {
    LAPACK_zgelss(&M, &N, &NRHS, A, &LDA, B, &LDB, S, &RCOND, &RANK, WORK, &LWORK, RWORK, &INFO);
  }

  void gesvd(char JOBU, char JOBVT, int M, int N, double *A, int LDA, double *S, double *U, int LDU, double *VT, int LDVT, double *WORK, int LWORK,
             [[maybe_unused]] double *RWORK, int &INFO) {
    LAPACK_dgesvd(&JOBU, &JOBVT, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT, WORK, &LWORK, &INFO);
  }
  void gesvd(char JOBU, char JOBVT, int M, int N, std::complex<double> *A, int LDA, double *S, std::complex<double> *U, int LDU,
             std::complex<double> *VT, int LDVT, std::complex<double> *WORK, int LWORK, double *RWORK, int &INFO) {
    LAPACK_zgesvd(&JOBU, &JOBVT, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT, WORK, &LWORK, RWORK, &INFO);
  }

  void geqp3(int M, int N, double *A, int LDA, int *JPVT, double *TAU, double *WORK, int LWORK, [[maybe_unused]] double *RWORK, int &INFO) {
    LAPACK_dgeqp3(&M, &N, A, &LDA, JPVT, TAU, WORK, &LWORK, &INFO);
  }
  void geqp3(int M, int N, std::complex<double> *A, int LDA, int *JPVT, std::complex<double> *TAU, std::complex<double> *WORK, int LWORK,
             double *RWORK, int &INFO) {
    LAPACK_zgeqp3(&M, &N, A, &LDA, JPVT, TAU, WORK, &LWORK, RWORK, &INFO);
  }

  void orgqr(int M, int N, int K, double *A, int LDA, double const *TAU, double *WORK, int LWORK, int &INFO) {
    LAPACK_dorgqr(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);
  }

  void ungqr(int M, int N, int K, std::complex<double> *A, int LDA, std::complex<double> const *TAU, std::complex<double> *WORK, int LWORK,
             int &INFO) {
    LAPACK_zungqr(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);
  }

  void getrf(int M, int N, double *A, int LDA, int *ipiv, int &info) { LAPACK_dgetrf(&M, &N, A, &LDA, ipiv, &info); }
  void getrf(int M, int N, std::complex<double> *A, int LDA, int *ipiv, int &info) { LAPACK_zgetrf(&M, &N, A, &LDA, ipiv, &info); }

  void getri(int N, double *A, int LDA, int const *ipiv, double *work, int lwork, int &info) {
    LAPACK_dgetri(&N, A, &LDA, ipiv, work, &lwork, &info);
  }
  void getri(int N, std::complex<double> *A, int LDA, int const *ipiv, std::complex<double> *work, int lwork, int &info) {
    LAPACK_zgetri(&N, A, &LDA, ipiv, work, &lwork, &info);
  }

  void gtsv(int N, int NRHS, double *DL, double *D, double *DU, double *B, int LDB, int &info) { LAPACK_dgtsv(&N, &NRHS, DL, D, DU, B, &LDB, &info); }
  void gtsv(int N, int NRHS, std::complex<double> *DL, std::complex<double> *D, std::complex<double> *DU, std::complex<double> *B, int LDB,
            int &info) {
    LAPACK_zgtsv(&N, &NRHS, DL, D, DU, B, &LDB, &info);
  }

  void stev(char J, int N, double *D, double *E, double *Z, int ldz, double *work, int &info) { LAPACK_dstev(&J, &N, D, E, Z, &ldz, work, &info); }

  void syev(char JOBZ, char UPLO, int N, double *A, int LDA, double *W, double *work, int lwork, int &info) {
    LAPACK_dsyev(&JOBZ, &UPLO, &N, A, &LDA, W, work, &lwork, &info);
  }

  void heev(char JOBZ, char UPLO, int N, std::complex<double> *A, int LDA, double *W, std::complex<double> *work, int lwork, double *rwork,
            int &info) {
    LAPACK_zheev(&JOBZ, &UPLO, &N, A, &LDA, W, work, &lwork, rwork, &info);
  }

  void sygv(int ITYPE, char JOBZ, char UPLO, int N, double *A, int LDA, double *B, int LDB, double *W, double *work, int lwork, int &info) {
    LAPACK_dsygv(&ITYPE, &JOBZ, &UPLO, &N, A, &LDA, B, &LDB, W, work, &lwork, &info);
  }

  void hegv(int ITYPE, char JOBZ, char UPLO, int N, std::complex<double> *A, int LDA, std::complex<double> *B, int LDB, double *W,
            std::complex<double> *work, int lwork, double *rwork, int &info) {
    LAPACK_zhegv(&ITYPE, &JOBZ, &UPLO, &N, A, &LDA, B, &LDB, W, work, &lwork, rwork, &info);
  }

  void getrs(char op, int N, int NRHS, double const *A, int LDA, int const *ipiv, double *B, int LDB, int &info) {
    LAPACK_dgetrs(&op, &N, &NRHS, A, &LDA, ipiv, B, &LDB, &info);
  }
  void getrs(char op, int N, int NRHS, std::complex<double> const *A, int LDA, int const *ipiv, std::complex<double> *B, int LDB, int &info) {
    LAPACK_zgetrs(&op, &N, &NRHS, A, &LDA, ipiv, B, &LDB, &info);
  }

  void geev(char JOBVL, char JOBVR, int N, double *A, int LDA, double *WR, double *WI, double *VL, int LDVL, double *VR, int LDVR, double *work,
            int lwork, int &info) {
    LAPACK_dgeev(&JOBVL, &JOBVR, &N, A, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, work, &lwork, &info);
  }
  void geev(char JOBVL, char JOBVR, int N, std::complex<double> *A, int LDA, std::complex<double> *W, std::complex<double> *VL, int LDVL,
            std::complex<double> *VR, int LDVR, std::complex<double> *work, int lwork, double *rwork, int &info) {
    LAPACK_zgeev(&JOBVL, &JOBVR, &N, A, &LDA, W, VL, &LDVL, VR, &LDVR, work, &lwork, rwork, &info);
  }

} // namespace nda::lapack::f77
