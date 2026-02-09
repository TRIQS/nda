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

  void gelss(int m, int n, int nrhs, float *a, int lda, float *b, int ldb, float *s, float rcond, int &rank, float *work, int lwork,
             [[maybe_unused]] float *rwork, int &info) {
    LAPACK_sgelss(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, work, &lwork, &info);
  }
  void gelss(int m, int n, int nrhs, std::complex<float> *a, int lda, std::complex<float> *b, int ldb, float *s, float rcond, int &rank,
             std::complex<float> *work, int lwork, float *rwork, int &info) {
    LAPACK_cgelss(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, work, &lwork, rwork, &info);
  }
  void gelss(int m, int n, int nrhs, double *a, int lda, double *b, int ldb, double *s, double rcond, int &rank, double *work, int lwork,
             [[maybe_unused]] double *rwork, int &info) {
    LAPACK_dgelss(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, work, &lwork, &info);
  }
  void gelss(int m, int n, int nrhs, std::complex<double> *a, int lda, std::complex<double> *b, int ldb, double *s, double rcond, int &rank,
             std::complex<double> *work, int lwork, double *rwork, int &info) {
    LAPACK_zgelss(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, work, &lwork, rwork, &info);
  }

  void gesvd(char jobu, char jobvt, int m, int n, float *a, int lda, float *s, float *u, int ldu, float *vt, int ldvt, float *work, int lwork,
             [[maybe_unused]] float *rwork, int &info) {
    LAPACK_sgesvd(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
  }
  void gesvd(char jobu, char jobvt, int m, int n, std::complex<float> *a, int lda, float *s, std::complex<float> *u, int ldu, std::complex<float> *vt,
             int ldvt, std::complex<float> *work, int lwork, float *rwork, int &info) {
    LAPACK_cgesvd(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, &info);
  }
  void gesvd(char jobu, char jobvt, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt, double *work, int lwork,
             [[maybe_unused]] double *rwork, int &info) {
    LAPACK_dgesvd(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
  }
  void gesvd(char jobu, char jobvt, int m, int n, std::complex<double> *a, int lda, double *s, std::complex<double> *u, int ldu,
             std::complex<double> *vt, int ldvt, std::complex<double> *work, int lwork, double *rwork, int &info) {
    LAPACK_zgesvd(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, &info);
  }

  void geqp3(int m, int n, float *a, int lda, int *jpvt, float *tau, float *work, int lwork, [[maybe_unused]] float *rwork, int &info) {
    LAPACK_sgeqp3(&m, &n, a, &lda, jpvt, tau, work, &lwork, &info);
  }
  void geqp3(int m, int n, std::complex<float> *a, int lda, int *jpvt, std::complex<float> *tau, std::complex<float> *work, int lwork, float *rwork,
             int &info) {
    LAPACK_cgeqp3(&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork, &info);
  }
  void geqp3(int m, int n, double *a, int lda, int *jpvt, double *tau, double *work, int lwork, [[maybe_unused]] double *rwork, int &info) {
    LAPACK_dgeqp3(&m, &n, a, &lda, jpvt, tau, work, &lwork, &info);
  }
  void geqp3(int m, int n, std::complex<double> *a, int lda, int *jpvt, std::complex<double> *tau, std::complex<double> *work, int lwork,
             double *rwork, int &info) {
    LAPACK_zgeqp3(&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork, &info);
  }

  void geqrf(int m, int n, float *a, int lda, float *tau, float *work, int lwork, int &info) {
    LAPACK_sgeqrf(&m, &n, a, &lda, tau, work, &lwork, &info);
  }
  void geqrf(int m, int n, std::complex<float> *a, int lda, std::complex<float> *tau, std::complex<float> *work, int lwork, int &info) {
    LAPACK_cgeqrf(&m, &n, a, &lda, tau, work, &lwork, &info);
  }
  void geqrf(int m, int n, double *a, int lda, double *tau, double *work, int lwork, int &info) {
    LAPACK_dgeqrf(&m, &n, a, &lda, tau, work, &lwork, &info);
  }
  void geqrf(int m, int n, std::complex<double> *a, int lda, std::complex<double> *tau, std::complex<double> *work, int lwork, int &info) {
    LAPACK_zgeqrf(&m, &n, a, &lda, tau, work, &lwork, &info);
  }

  void orgqr(int m, int n, int k, float *a, int lda, float const *tau, float *work, int lwork, int &info) {
    LAPACK_sorgqr(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  }
  void orgqr(int m, int n, int k, double *a, int lda, double const *tau, double *work, int lwork, int &info) {
    LAPACK_dorgqr(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  }

  void ungqr(int m, int n, int k, std::complex<float> *a, int lda, std::complex<float> const *tau, std::complex<float> *work, int lwork, int &info) {
    LAPACK_cungqr(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  }
  void ungqr(int m, int n, int k, std::complex<double> *a, int lda, std::complex<double> const *tau, std::complex<double> *work, int lwork,
             int &info) {
    LAPACK_zungqr(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  }

  void getrf(int m, int n, float *a, int lda, int *ipiv, int &info) { LAPACK_sgetrf(&m, &n, a, &lda, ipiv, &info); }
  void getrf(int m, int n, std::complex<float> *a, int lda, int *ipiv, int &info) { LAPACK_cgetrf(&m, &n, a, &lda, ipiv, &info); }
  void getrf(int m, int n, double *a, int lda, int *ipiv, int &info) { LAPACK_dgetrf(&m, &n, a, &lda, ipiv, &info); }
  void getrf(int m, int n, std::complex<double> *a, int lda, int *ipiv, int &info) { LAPACK_zgetrf(&m, &n, a, &lda, ipiv, &info); }

  void getri(int n, float *a, int lda, int const *ipiv, float *work, int lwork, int &info) { LAPACK_sgetri(&n, a, &lda, ipiv, work, &lwork, &info); }
  void getri(int n, std::complex<float> *a, int lda, int const *ipiv, std::complex<float> *work, int lwork, int &info) {
    LAPACK_cgetri(&n, a, &lda, ipiv, work, &lwork, &info);
  }
  void getri(int n, double *a, int lda, int const *ipiv, double *work, int lwork, int &info) {
    LAPACK_dgetri(&n, a, &lda, ipiv, work, &lwork, &info);
  }
  void getri(int n, std::complex<double> *a, int lda, int const *ipiv, std::complex<double> *work, int lwork, int &info) {
    LAPACK_zgetri(&n, a, &lda, ipiv, work, &lwork, &info);
  }

  void gtsv(int n, int nrhs, double *dl, double *d, double *du, double *b, int ldb, int &info) { LAPACK_dgtsv(&n, &nrhs, dl, d, du, b, &ldb, &info); }
  void gtsv(int n, int nrhs, std::complex<double> *dl, std::complex<double> *d, std::complex<double> *du, std::complex<double> *b, int ldb,
            int &info) {
    LAPACK_zgtsv(&n, &nrhs, dl, d, du, b, &ldb, &info);
  }

  void stev(char j, int n, double *d, double *e, double *z, int ldz, double *work, int &info) { LAPACK_dstev(&j, &n, d, e, z, &ldz, work, &info); }

  void syev(char jobz, char uplo, int n, double *a, int lda, double *w, double *work, int lwork, int &info) {
    LAPACK_dsyev(&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info);
  }

  void heev(char jobz, char uplo, int n, std::complex<double> *a, int lda, double *w, std::complex<double> *work, int lwork, double *rwork,
            int &info) {
    LAPACK_zheev(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
  }

  void sygv(int itype, char jobz, char uplo, int n, double *a, int lda, double *b, int ldb, double *w, double *work, int lwork, int &info) {
    LAPACK_dsygv(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, &info);
  }

  void hegv(int itype, char jobz, char uplo, int n, std::complex<double> *a, int lda, std::complex<double> *b, int ldb, double *w,
            std::complex<double> *work, int lwork, double *rwork, int &info) {
    LAPACK_zhegv(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &info);
  }

  void getrs(char op, int n, int nrhs, float const *a, int lda, int const *ipiv, float *b, int ldb, int &info) {
    LAPACK_sgetrs(&op, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  }
  void getrs(char op, int n, int nrhs, std::complex<float> const *a, int lda, int const *ipiv, std::complex<float> *b, int ldb, int &info) {
    LAPACK_cgetrs(&op, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  }
  void getrs(char op, int n, int nrhs, double const *a, int lda, int const *ipiv, double *b, int ldb, int &info) {
    LAPACK_dgetrs(&op, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  }
  void getrs(char op, int n, int nrhs, std::complex<double> const *a, int lda, int const *ipiv, std::complex<double> *b, int ldb, int &info) {
    LAPACK_zgetrs(&op, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  }

  void geev(char jobvl, char jobvr, int n, double *a, int lda, double *wr, double *wi, double *vl, int ldvl, double *vr, int ldvr, double *work,
            int lwork, int &info) {
    LAPACK_dgeev(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
  }
  void geev(char jobvl, char jobvr, int n, std::complex<double> *a, int lda, std::complex<double> *w, std::complex<double> *vl, int ldvl,
            std::complex<double> *vr, int ldvr, std::complex<double> *work, int lwork, double *rwork, int &info) {
    LAPACK_zgeev(&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work, &lwork, rwork, &info);
  }

} // namespace nda::lapack::f77
