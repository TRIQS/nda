// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a C++ interface for various LAPACK routines.
 */

#pragma once

#include "./cusolver_interface.hpp"
#include "../../blas/tools.hpp"

#include <complex>

namespace nda::lapack::f77 {

  void gelss(int m, int n, int nrhs, float *a, int lda, float *b, int ldb, float *s, float rcond, int &rank, float *work, int lwork, float *rwork,
             int &info);
  void gelss(int m, int n, int nrhs, std::complex<float> *a, int lda, std::complex<float> *b, int ldb, float *s, float rcond, int &rank,
             std::complex<float> *work, int lwork, float *rwork, int &info);
  void gelss(int m, int n, int nrhs, double *a, int lda, double *b, int ldb, double *s, double rcond, int &rank, double *work, int lwork,
             double *rwork, int &info);
  void gelss(int m, int n, int nrhs, std::complex<double> *a, int lda, std::complex<double> *b, int ldb, double *s, double rcond, int &rank,
             std::complex<double> *work, int lwork, double *rwork, int &info);

  void gesvd(char jobu, char jobvt, int m, int n, float *a, int lda, float *s, float *u, int ldu, float *vt, int ldvt, float *work, int lwork,
             float *rwork, int &info);
  void gesvd(char jobu, char jobvt, int m, int n, std::complex<float> *a, int lda, float *s, std::complex<float> *u, int ldu, std::complex<float> *vt,
             int ldvt, std::complex<float> *work, int lwork, float *rwork, int &info);
  void gesvd(char jobu, char jobvt, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt, double *work, int lwork,
             double *rwork, int &info);
  void gesvd(char jobu, char jobvt, int m, int n, std::complex<double> *a, int lda, double *s, std::complex<double> *u, int ldu,
             std::complex<double> *vt, int ldvt, std::complex<double> *work, int lwork, double *rwork, int &info);

  void geqp3(int m, int n, float *a, int lda, int *jpvt, float *tau, float *work, int lwork, float *rwork, int &info);
  void geqp3(int m, int n, std::complex<float> *a, int lda, int *jpvt, std::complex<float> *tau, std::complex<float> *work, int lwork, float *rwork,
             int &info);
  void geqp3(int m, int n, double *a, int lda, int *jpvt, double *tau, double *work, int lwork, double *rwork, int &info);
  void geqp3(int m, int n, std::complex<double> *a, int lda, int *jpvt, std::complex<double> *tau, std::complex<double> *work, int lwork,
             double *rwork, int &info);

  void geqrf(int m, int n, float *a, int lda, float *tau, float *work, int lwork, int &info);
  void geqrf(int m, int n, std::complex<float> *a, int lda, std::complex<float> *tau, std::complex<float> *work, int lwork, int &info);
  void geqrf(int m, int n, double *a, int lda, double *tau, double *work, int lwork, int &info);
  void geqrf(int m, int n, std::complex<double> *a, int lda, std::complex<double> *tau, std::complex<double> *work, int lwork, int &info);

  void orgqr(int m, int n, int k, float *a, int lda, float const *tau, float *work, int lwork, int &info);
  void orgqr(int m, int n, int k, double *a, int lda, double const *tau, double *work, int lwork, int &info);

  void ungqr(int m, int n, int k, std::complex<float> *a, int lda, std::complex<float> const *tau, std::complex<float> *work, int lwork, int &info);
  void ungqr(int m, int n, int k, std::complex<double> *a, int lda, std::complex<double> const *tau, std::complex<double> *work, int lwork,
             int &info);

  void getrf(int m, int n, float *a, int lda, int *ipiv, int &info);
  void getrf(int m, int n, std::complex<float> *a, int lda, int *ipiv, int &info);
  void getrf(int m, int n, double *a, int lda, int *ipiv, int &info);
  void getrf(int m, int n, std::complex<double> *a, int lda, int *ipiv, int &info);

  void getri(int n, float *a, int lda, int const *ipiv, float *work, int lwork, int &info);
  void getri(int n, std::complex<float> *a, int lda, int const *ipiv, std::complex<float> *work, int lwork, int &info);
  void getri(int n, double *a, int lda, int const *ipiv, double *work, int lwork, int &info);
  void getri(int n, std::complex<double> *a, int lda, int const *ipiv, std::complex<double> *work, int lwork, int &info);

  void gtsv(int n, int nrhs, double *dl, double *d, double *du, double *b, int ldb, int &info);
  void gtsv(int n, int nrhs, std::complex<double> *dl, std::complex<double> *d, std::complex<double> *du, std::complex<double> *b, int ldb,
            int &info);

  void stev(char j, int n, double *d, double *e, double *z, int ldz, double *work, int &info);

  void syev(char jobz, char uplo, int n, double *a, int lda, double *w, double *work, int lwork, int &info);

  void heev(char jobz, char uplo, int n, std::complex<double> *a, int lda, double *w, std::complex<double> *work, int lwork, double *rwork,
            int &info);

  void sygv(int itype, char jobz, char uplo, int n, double *a, int lda, double *b, int ldb, double *w, double *work, int lwork, int &info);

  void hegv(int itype, char jobz, char uplo, int n, std::complex<double> *a, int lda, std::complex<double> *b, int ldb, double *w,
            std::complex<double> *work, int lwork, double *rwork, int &info);

  void getrs(char op, int n, int nrhs, float const *a, int lda, int const *ipiv, float *b, int ldb, int &info);
  void getrs(char op, int n, int nrhs, std::complex<float> const *a, int lda, int const *ipiv, std::complex<float> *b, int ldb, int &info);
  void getrs(char op, int n, int nrhs, double const *a, int lda, int const *ipiv, double *b, int ldb, int &info);
  void getrs(char op, int n, int nrhs, std::complex<double> const *a, int lda, int const *ipiv, std::complex<double> *b, int ldb, int &info);

  void geev(char jobvl, char jobvr, int n, double *a, int lda, double *wr, double *wi, double *vl, int ldvl, double *vr, int ldvr, double *work,
            int lwork, int &info);
  void geev(char jobvl, char jobvr, int n, std::complex<double> *a, int lda, std::complex<double> *w, std::complex<double> *vl, int ldvl,
            std::complex<double> *vr, int ldvr, std::complex<double> *work, int lwork, double *rwork, int &info);

} // namespace nda::lapack::f77

namespace nda::lapack {

  // Import tools from the blas_lapack namespace.
  using namespace nda::blas_lapack;

} // namespace nda::lapack
