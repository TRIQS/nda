// Copyright (c) 2019-2021 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Olivier Parcollet, Nils Wentzell

#include <nda/nda.hpp>
#include <nda/macros.hpp>
#include <nda/exceptions.hpp>
#include <nda/mem/handle.hpp>
#include "lapack_cxx_interface.hpp"

// Extracted from Reference Lapack (https://github.com/Reference-LAPACK):
#include "lapack.h"

#include <string>

using namespace std::string_literals;

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

  void getrf(int M, int N, double *A, int LDA, int *ipiv, int &info) { LAPACK_dgetrf(&M, &N, A, &LDA, ipiv, &info); }
  void getrf(int M, int N, std::complex<double> *A, int LDA, int *ipiv, int &info) { LAPACK_zgetrf(&M, &N, A, &LDA, ipiv, &info); }

  void getri(int N, double *A, int LDA, int const *ipiv, double *work, int lwork, int &info) { LAPACK_dgetri(&N, A, &LDA, ipiv, work, &lwork, &info); }
  void getri(int N, std::complex<double> *A, int LDA, int const *ipiv, std::complex<double> *work, int lwork, int &info) {
    LAPACK_zgetri(&N, A, &LDA, ipiv, work, &lwork, &info);
  }

  void gtsv(int N, int NRHS, double *DL, double *D, double *DU, double *B, int LDB, int &info) { LAPACK_dgtsv(&N, &NRHS, DL, D, DU, B, &LDB, &info); }
  void gtsv(int N, int NRHS, std::complex<double> *DL, std::complex<double> *D, std::complex<double> *DU, std::complex<double> *B, int LDB,
            int &info) {
    LAPACK_zgtsv(&N, &NRHS, DL, D, DU, B, &LDB, &info);
  }

  void stev(char J, int N, double *D, double *E, double *Z, int ldz, double *work, int &info) { LAPACK_dstev(&J, &N, D, E, Z, &ldz, work, &info); }

  void syev(char JOBZ, char UPLO, int N, double *A, int LDA, double *W, double *work, int &lwork, int &info) {
    LAPACK_dsyev(&JOBZ, &UPLO, &N, A, &LDA, W, work, &lwork, &info);
  }

  void heev(char JOBZ, char UPLO, int N, std::complex<double> *A, int LDA, double *W, std::complex<double> *work, int &lwork, double *work2,
            int &info) {
    LAPACK_zheev(&JOBZ, &UPLO, &N, A, &LDA, W, work, &lwork, work2, &info);
  }

  void getrs(char op, int N, int NRHS, double const *A, int LDA, int const *ipiv, double *B, int LDB, int &info) {
    LAPACK_dgetrs(&op, &N, &NRHS, A, &LDA, ipiv, B, &LDB, &info);
  }
  void getrs(char op, int N, int NRHS, std::complex<double> const *A, int LDA, int const *ipiv, std::complex<double> *B, int LDB, int &info) {
    LAPACK_zgetrs(&op, &N, &NRHS, A, &LDA, ipiv, B, &LDB, &info);
  }

} // namespace nda::lapack::f77

