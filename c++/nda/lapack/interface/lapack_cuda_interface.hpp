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

#pragma once

#include <complex>

namespace nda::lapack::device {

  void gesvd(char JOBU, char JOBVT, int M, int N, double *A, int LDA, double *S, double *U, int LDU, double *VT, int LDVT, double *WORK, int LWORK,
             double *RWORK, int &INFO);
  void gesvd(char JOBU, char JOBVT, int M, int N, std::complex<double> *A, int LDA, double *S, std::complex<double> *U, int LDU,
             std::complex<double> *VT, int LDVT, std::complex<double> *WORK, int LWORK, double *RWORK, int &INFO);

  void getrf(int M, int N, double *A, int LDA, int *ipiv, int &info);
  void getrf(int M, int N, std::complex<double> *A, int LDA, int *ipiv, int &info);

  void getri(int N, double *A, int LDA, int *ipiv, double *WORK, int LWORK, int &info);
  void getri(int N, std::complex<double> *A, int LDA, int *ipiv, std::complex<double> *WORK, int LWORK, int &info);

  void getrs(char op, int N, int NRHS, double const *A, int LDA, int const *ipiv, double *B, int LDB, int &info);
  void getrs(char op, int N, int NRHS, std::complex<double> const *A, int LDA, int const *ipiv, std::complex<double> *B, int LDB, int &info);

} // namespace nda::lapack::device
