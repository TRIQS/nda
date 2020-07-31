// Copyright (c) 2019 Simons Foundation
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

#pragma once

#include <complex>

namespace nda::lapack::f77 {

  void gelss(int M, int N, int NRHS, double *A, int LDA, double *B, int LDB, double *S, double RCOND, int &RANK, double *WORK, int LWORK, int &INFO);
  void gelss(int M, int N, int NRHS, std::complex<double> *A, int LDA, std::complex<double> *B, int LDB, double *S, double RCOND, int &RANK,
             std::complex<double> *WORK, int LWORK, double *RWORK, int &INFO);

  void gesvd(const char &JOBU, const char &JOBVT, int M, int N, double *A, int LDA, double *S, double *U, int LDU, double *VT, int LDVT, double *WORK,
             int LWORK, int &INFO);
  void gesvd(const char &JOBU, const char &JOBVT, int M, int N, std::complex<double> *A, int LDA, double *S, std::complex<double> *U, int LDU,
             std::complex<double> *VT, int LDVT, std::complex<double> *WORK, int LWORK, double *RWORK, int &INFO);

  void getrf(int M, int N, double *A, int LDA, int *ipiv, int &info);
  void getrf(int M, int N, std::complex<double> *A, int LDA, int *ipiv, int &info);

  void getri(int N, double *A, int LDA, int *ipiv, double *work, int lwork, int &info);
  void getri(int N, std::complex<double> *A, int LDA, int *ipiv, std::complex<double> *work, int lwork, int &info);

  void gtsv(int N, int NRHS, double *DL, double *D, double *DU, double *B, int LDB, int &info);
  void gtsv(int N, int NRHS, std::complex<double> *DL, std::complex<double> *D, std::complex<double> *DU, std::complex<double> *B, int LDB,
            int &info);

  void stev(char J, int N, double *D, double *E, double *Z, int ldz, double *work, int &info);

  void dsyev(char JOBZ, char UPLO, int N, double *A, int LDA, double *W, double *work, int &lwork, int &info);
  void zheev(char JOBZ, char UPLO, int N, std::complex<double> *A, int LDA, double *W, std::complex<double> *work, int &lwork, double *work2,
             int &info);
} // namespace nda::lapack::f77
