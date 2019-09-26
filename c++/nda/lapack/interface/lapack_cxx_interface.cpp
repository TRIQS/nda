/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011-2017 by O. Parcollet
 * Copyright (C) 2018 by Simons Foundation
 *   author : O. Parcollet, P. Dumitrescu, N. Wentzell
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include "lapack_cxx_interface.hpp"
// Extracted from Reference Lapack (https://github.com/Reference-LAPACK):
#include "lapack.h"

namespace nda::lapack::f77 {

  void gelss(int M, int N, int NRHS, double *A, int LDA, double *B, int LDB, double *S, double RCOND, int &RANK, double *WORK, int LWORK, int &INFO) {
    LAPACK_dgelss(&M, &N, &NRHS, A, &LDA, B, &LDB, S, &RCOND, &RANK, WORK, &LWORK, &INFO);
  }
  void gelss(int M, int N, int NRHS, std::complex<double> *A, int LDA, std::complex<double> *B, int LDB, double *S, double RCOND, int &RANK,
             std::complex<double> *WORK, int LWORK, double *RWORK, int &INFO) {
    LAPACK_zgelss(&M, &N, &NRHS, A, &LDA, B, &LDB, S, &RCOND, &RANK, WORK, &LWORK, RWORK, &INFO);
  }

  void gesvd(const char &JOBU, const char &JOBVT, int M, int N, double *A, int LDA, double *S, double *U, int LDU, double *VT, int LDVT, double *WORK,
             int LWORK, int &INFO) {
    LAPACK_dgesvd(&JOBU, &JOBVT, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT, WORK, &LWORK, &INFO);
  }
  void gesvd(const char &JOBU, const char &JOBVT, int M, int N, std::complex<double> *A, int LDA, double *S, std::complex<double> *U, int LDU,
             std::complex<double> *VT, int LDVT, std::complex<double> *WORK, int LWORK, double *RWORK, int &INFO) {
    LAPACK_zgesvd(&JOBU, &JOBVT, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT, WORK, &LWORK, RWORK, &INFO);
  }

  void getrf(int M, int N, double *A, int LDA, int *ipiv, int &info) { LAPACK_dgetrf(&M, &N, A, &LDA, ipiv, &info); }
  void getrf(int M, int N, std::complex<double> *A, int LDA, int *ipiv, int &info) { LAPACK_zgetrf(&M, &N, A, &LDA, ipiv, &info); }

  void getri(int N, double *A, int LDA, int *ipiv, double *work, int lwork, int &info) { LAPACK_dgetri(&N, A, &LDA, ipiv, work, &lwork, &info); }
  void getri(int N, std::complex<double> *A, int LDA, int *ipiv, std::complex<double> *work, int lwork, int &info) {
    LAPACK_zgetri(&N, A, &LDA, ipiv, work, &lwork, &info);
  }

  void gtsv(int N, int NRHS, double *DL, double *D, double *DU, double *B, int LDB, int &info) { LAPACK_dgtsv(&N, &NRHS, DL, D, DU, B, &LDB, &info); }
  void gtsv(int N, int NRHS, std::complex<double> *DL, std::complex<double> *D, std::complex<double> *DU, std::complex<double> *B, int LDB,
            int &info) {
    LAPACK_zgtsv(&N, &NRHS, DL, D, DU, B, &LDB, &info);
  }

  void stev(char J, int N, double *D, double *E, double *Z, int ldz, double *work, int &info) { LAPACK_dstev(&J, &N, D, E, Z, &ldz, work, &info); }

  void dsyev(char JOBZ, char UPLO, int N, double *A, int LDA, double *W, double *work, int &lwork, int &info) {
    LAPACK_dsyev(&JOBZ, &UPLO, &N, A, &LDA, W, work, &lwork, &info);
  }
  void zheev(char JOBZ, char UPLO, int N, std::complex<double> *A, int LDA, double *W, std::complex<double> *work, int &lwork, double *work2,
             int &info) {
    LAPACK_zheev(&JOBZ, &UPLO, &N, A, &LDA, W, work, &lwork, work2, &info);
  }

} // namespace nda::lapack::f77
