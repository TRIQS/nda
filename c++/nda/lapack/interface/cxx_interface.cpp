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

  void gelss(int M, int N, int NRHS, float *A, int LDA, float *B, int LDB, float *S, float RCOND, int &RANK, float *WORK, int LWORK,
             [[maybe_unused]] float *RWORK, int &INFO) {
    LAPACK_sgelss(&M, &N, &NRHS, A, &LDA, B, &LDB, S, &RCOND, &RANK, WORK, &LWORK, &INFO);
  }
  void gelss(int M, int N, int NRHS, double *A, int LDA, double *B, int LDB, double *S, double RCOND, int &RANK, double *WORK, int LWORK,
             [[maybe_unused]] double *RWORK, int &INFO) {
    LAPACK_dgelss(&M, &N, &NRHS, A, &LDA, B, &LDB, S, &RCOND, &RANK, WORK, &LWORK, &INFO);
  }
  void gelss(int M, int N, int NRHS, std::complex<float> *A, int LDA, std::complex<float> *B, int LDB, float *S, float RCOND, int &RANK,
             std::complex<float> *WORK, int LWORK, float *RWORK, int &INFO) {
    LAPACK_cgelss(&M, &N, &NRHS, A, &LDA, B, &LDB, S, &RCOND, &RANK, WORK, &LWORK, RWORK, &INFO);
  }
  void gelss(int M, int N, int NRHS, std::complex<double> *A, int LDA, std::complex<double> *B, int LDB, double *S, double RCOND, int &RANK,
             std::complex<double> *WORK, int LWORK, double *RWORK, int &INFO) {
    LAPACK_zgelss(&M, &N, &NRHS, A, &LDA, B, &LDB, S, &RCOND, &RANK, WORK, &LWORK, RWORK, &INFO);
  }

  void gesvd(char JOBU, char JOBVT, int M, int N, float *A, int LDA, float *S, float *U, int LDU, float *VT, int LDVT, float *WORK, int LWORK,
             [[maybe_unused]] float *RWORK, int &INFO) {
    LAPACK_sgesvd(&JOBU, &JOBVT, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT, WORK, &LWORK, &INFO);
  }
  void gesvd(char JOBU, char JOBVT, int M, int N, double *A, int LDA, double *S, double *U, int LDU, double *VT, int LDVT, double *WORK, int LWORK,
             [[maybe_unused]] double *RWORK, int &INFO) {
    LAPACK_dgesvd(&JOBU, &JOBVT, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT, WORK, &LWORK, &INFO);
  }
  void gesvd(char JOBU, char JOBVT, int M, int N, std::complex<float> *A, int LDA, float *S, std::complex<float> *U, int LDU, std::complex<float> *VT,
             int LDVT, std::complex<float> *WORK, int LWORK, float *RWORK, int &INFO) {
    LAPACK_cgesvd(&JOBU, &JOBVT, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT, WORK, &LWORK, RWORK, &INFO);
  }
  void gesvd(char JOBU, char JOBVT, int M, int N, std::complex<double> *A, int LDA, double *S, std::complex<double> *U, int LDU,
             std::complex<double> *VT, int LDVT, std::complex<double> *WORK, int LWORK, double *RWORK, int &INFO) {
    LAPACK_zgesvd(&JOBU, &JOBVT, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT, WORK, &LWORK, RWORK, &INFO);
  }

  void geqp3(int M, int N, float *A, int LDA, int *JPVT, float *TAU, float *WORK, int LWORK, [[maybe_unused]] float *RWORK, int &INFO) {
    LAPACK_sgeqp3(&M, &N, A, &LDA, JPVT, TAU, WORK, &LWORK, &INFO);
  }
  void geqp3(int M, int N, double *A, int LDA, int *JPVT, double *TAU, double *WORK, int LWORK, [[maybe_unused]] double *RWORK, int &INFO) {
    LAPACK_dgeqp3(&M, &N, A, &LDA, JPVT, TAU, WORK, &LWORK, &INFO);
  }
  void geqp3(int M, int N, std::complex<float> *A, int LDA, int *JPVT, std::complex<float> *TAU, std::complex<float> *WORK, int LWORK, float *RWORK,
             int &INFO) {
    LAPACK_cgeqp3(&M, &N, A, &LDA, JPVT, TAU, WORK, &LWORK, RWORK, &INFO);
  }
  void geqp3(int M, int N, std::complex<double> *A, int LDA, int *JPVT, std::complex<double> *TAU, std::complex<double> *WORK, int LWORK,
             double *RWORK, int &INFO) {
    LAPACK_zgeqp3(&M, &N, A, &LDA, JPVT, TAU, WORK, &LWORK, RWORK, &INFO);
  }

#define _geqrf_(FUN, TYPE)                                                                                                                           \
  void geqrf(int M, int N, TYPE *A, int LDA, TYPE *TAU, TYPE *WORK, int LWORK, int &INFO) { FUN(&M, &N, A, &LDA, TAU, WORK, &LWORK, &INFO); }
  _geqrf_(LAPACK_sgeqrf, float) _geqrf_(LAPACK_dgeqrf, double) _geqrf_(LAPACK_cgeqrf, std::complex<float>)
     _geqrf_(LAPACK_zgeqrf, std::complex<double>)

        void orgqr(int M, int N, int K, float *A, int LDA, float *TAU, float *WORK, int LWORK, int &INFO) {
    LAPACK_sorgqr(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);
  }
  void orgqr(int M, int N, int K, double *A, int LDA, double *TAU, double *WORK, int LWORK, int &INFO) {
    LAPACK_dorgqr(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);
  }

  void ungqr(int M, int N, int K, std::complex<float> *A, int LDA, std::complex<float> *TAU, std::complex<float> *WORK, int LWORK, int &INFO) {
    LAPACK_cungqr(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);
  }
  void ungqr(int M, int N, int K, std::complex<double> *A, int LDA, std::complex<double> *TAU, std::complex<double> *WORK, int LWORK, int &INFO) {
    LAPACK_zungqr(&M, &N, &K, A, &LDA, TAU, WORK, &LWORK, &INFO);
  }

#define _getrf_(FUN, TYPE)                                                                                                                           \
  void getrf(int M, int N, TYPE *A, int LDA, int *ipiv, int &info) { FUN(&M, &N, A, &LDA, ipiv, &info); }
  _getrf_(LAPACK_sgetrf, float) _getrf_(LAPACK_dgetrf, double) _getrf_(LAPACK_cgetrf, std::complex<float>)
     _getrf_(LAPACK_zgetrf, std::complex<double>)

#define _getri_(FUN, TYPE)                                                                                                                           \
  void getri(int N, TYPE *A, int LDA, int const *ipiv, TYPE *work, int lwork, int &info) { FUN(&N, A, &LDA, ipiv, work, &lwork, &info); }
        _getri_(LAPACK_sgetri, float) _getri_(LAPACK_dgetri, double) _getri_(LAPACK_cgetri, std::complex<float>)
           _getri_(LAPACK_zgetri, std::complex<double>)

              void gtsv(int N, int NRHS, double *DL, double *D, double *DU, double *B, int LDB, int &info) {
    LAPACK_dgtsv(&N, &NRHS, DL, D, DU, B, &LDB, &info);
  }
  void gtsv(int N, int NRHS, std::complex<double> *DL, std::complex<double> *D, std::complex<double> *DU, std::complex<double> *B, int LDB,
            int &info) {
    LAPACK_zgtsv(&N, &NRHS, DL, D, DU, B, &LDB, &info);
  }

  void stev(char J, int N, double *D, double *E, double *Z, int ldz, double *work, int &info) { LAPACK_dstev(&J, &N, D, E, Z, &ldz, work, &info); }

  void syev(char JOBZ, char UPLO, int N, float *A, int LDA, float *W, float *work, int &lwork, int &info) {
    LAPACK_ssyev(&JOBZ, &UPLO, &N, A, &LDA, W, work, &lwork, &info);
  }
  void syev(char JOBZ, char UPLO, int N, double *A, int LDA, double *W, double *work, int &lwork, int &info) {
    LAPACK_dsyev(&JOBZ, &UPLO, &N, A, &LDA, W, work, &lwork, &info);
  }

  void heev(char JOBZ, char UPLO, int N, std::complex<float> *A, int LDA, float *W, std::complex<float> *work, int &lwork, float *work2, int &info) {
    LAPACK_cheev(&JOBZ, &UPLO, &N, A, &LDA, W, work, &lwork, work2, &info);
  }
  void heev(char JOBZ, char UPLO, int N, std::complex<double> *A, int LDA, double *W, std::complex<double> *work, int &lwork, double *work2,
            int &info) {
    LAPACK_zheev(&JOBZ, &UPLO, &N, A, &LDA, W, work, &lwork, work2, &info);
  }

  void sygv(int ITYPE, char JOBZ, char UPLO, int N, double *A, int LDA, double *B, int LDB, double *W, double *work, int &lwork, int &info) {
    LAPACK_dsygv(&ITYPE, &JOBZ, &UPLO, &N, A, &LDA, B, &LDB, W, work, &lwork, &info);
  }

  void hegv(int ITYPE, char JOBZ, char UPLO, int N, std::complex<double> *A, int LDA, std::complex<double> *B, int LDB, double *W,
            std::complex<double> *work, int &lwork, double *work2, int &info) {
    LAPACK_zhegv(&ITYPE, &JOBZ, &UPLO, &N, A, &LDA, B, &LDB, W, work, &lwork, work2, &info);
  }

  void geev(char JOBVL, char JOBVR, int N, float *A, int LDA, float *WR, float *WI, float *VL, int LDVL, float *VR, int LDVR, float *work, int &lwork,
            int &info) {
    LAPACK_sgeev(&JOBVL, &JOBVR, &N, A, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, work, &lwork, &info);
  }
  void geev(char JOBVL, char JOBVR, int N, double *A, int LDA, double *WR, double *WI, double *VL, int LDVL, double *VR, int LDVR, double *work,
            int &lwork, int &info) {
    LAPACK_dgeev(&JOBVL, &JOBVR, &N, A, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, work, &lwork, &info);
  }
  void geev(char JOBVL, char JOBVR, int N, std::complex<float> *A, int LDA, std::complex<float> *W, std::complex<float> *VL, int LDVL,
            std::complex<float> *VR, int LDVR, std::complex<float> *work, int &lwork, float *work2, int &info) {
    LAPACK_cgeev(&JOBVL, &JOBVR, &N, A, &LDA, W, VL, &LDVL, VR, &LDVR, work, &lwork, work2, &info);
  }
  void geev(char JOBVL, char JOBVR, int N, std::complex<double> *A, int LDA, std::complex<double> *W, std::complex<double> *VL, int LDVL,
            std::complex<double> *VR, int LDVR, std::complex<double> *work, int &lwork, double *work2, int &info) {
    LAPACK_zgeev(&JOBVL, &JOBVR, &N, A, &LDA, W, VL, &LDVL, VR, &LDVR, work, &lwork, work2, &info);
  }

  void getrs(char op, int N, int NRHS, float const *A, int LDA, int const *ipiv, float *B, int LDB, int &info) {
    LAPACK_sgetrs(&op, &N, &NRHS, A, &LDA, ipiv, B, &LDB, &info);
  }
  void getrs(char op, int N, int NRHS, std::complex<float> const *A, int LDA, int const *ipiv, std::complex<float> *B, int LDB, int &info) {
    LAPACK_cgetrs(&op, &N, &NRHS, A, &LDA, ipiv, B, &LDB, &info);
  }
  void getrs(char op, int N, int NRHS, double const *A, int LDA, int const *ipiv, double *B, int LDB, int &info) {
    LAPACK_dgetrs(&op, &N, &NRHS, A, &LDA, ipiv, B, &LDB, &info);
  }
  void getrs(char op, int N, int NRHS, std::complex<double> const *A, int LDA, int const *ipiv, std::complex<double> *B, int LDB, int &info) {
    LAPACK_zgetrs(&op, &N, &NRHS, A, &LDA, ipiv, B, &LDB, &info);
  }

} // namespace nda::lapack::f77
