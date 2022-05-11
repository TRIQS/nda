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
#include "cusolverDn.h"

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

namespace nda::lapack::cuda {

  constexpr cublasOperation_t get_cublas_op(char op) {
    switch (op) {
      case 'N': return CUBLAS_OP_N; break;
      case 'T': return CUBLAS_OP_T; break;
      case 'C': return CUBLAS_OP_C; break;
      default: std::terminate(); return {};
    }
  }

  struct handle_t {
    handle_t() { cusolverDnCreate(&h); }
    ~handle_t() { cusolverDnDestroy(h); }
    operator cusolverDnHandle_t() { return h; }

    private:
    cusolverDnHandle_t h = {};
  };
  static handle_t handle = {}; // NOLINT

#define CUSOLVER_CHECK(X, ...)                                                                                                                       \
  auto err = X(handle, __VA_ARGS__);                                                                                                                 \
  if (err != CUSOLVER_STATUS_SUCCESS) NDA_RUNTIME_ERROR << AS_STRING(X) + " failed with error code "s + std::to_string(err);

  inline auto *cucplx(std::complex<double> *c) { return reinterpret_cast<cuDoubleComplex *>(c); }             // NOLINT
  inline auto *cucplx(std::complex<double> const *c) { return reinterpret_cast<const cuDoubleComplex *>(c); } // NOLINT

  static auto info_u_handle = mem::handle_heap<int, mem::mallocator<mem::Unified>>(1); // NOLINT
  auto &info_u              = *info_u_handle.data();                                   // NOLINT

  void gesvd(char JOBU, char JOBVT, int M, int N, double *A, int LDA, double *S, double *U, int LDU, double *VT, int LDVT, double *WORK, int LWORK,
             double *RWORK, int &INFO) {
    // Replicate behavior of Netlib gesvd
    if (LWORK == -1) {
      int bufferSize = 0;
      cusolverDnDgesvd_bufferSize(handle, M, N, &bufferSize);
      *WORK = bufferSize;
    } else {
      CUSOLVER_CHECK(cusolverDnDgesvd, JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, RWORK, &info_u);
      INFO = info_u;
    }
  }
  void gesvd(char JOBU, char JOBVT, int M, int N, std::complex<double> *A, int LDA, double *S, std::complex<double> *U, int LDU,
             std::complex<double> *VT, int LDVT, std::complex<double> *WORK, int LWORK, double *RWORK, int &INFO) {
    // Replicate behavior of Netlib gesvd
    if (LWORK == -1) {
      int bufferSize = 0;
      cusolverDnZgesvd_bufferSize(handle, M, N, &bufferSize);
      *WORK = bufferSize;
    } else {
      CUSOLVER_CHECK(cusolverDnZgesvd, JOBU, JOBVT, M, N, cucplx(A), LDA, S, cucplx(U), LDU, cucplx(VT), LDVT, cucplx(WORK), LWORK, RWORK,
                     &info_u); // NOLINT
      INFO = info_u;
    }
  }

  void getrf(int M, int N, double *A, int LDA, int *ipiv, int &info) {
    int bufferSize = 0;
    cusolverDnDgetrf_bufferSize(handle, M, N, A, LDA, &bufferSize);
    auto Workspace = nda::cuvector<double>(bufferSize);
    CUSOLVER_CHECK(cusolverDnDgetrf, M, N, A, LDA, Workspace.data(), ipiv, &info_u);
    info = info_u;
  }
  void getrf(int M, int N, std::complex<double> *A, int LDA, int *ipiv, int &info) {
    int bufferSize = 0;
    cusolverDnZgetrf_bufferSize(handle, M, N, cucplx(A), LDA, &bufferSize);
    auto Workspace = nda::cuvector<dcomplex>(bufferSize);
    CUSOLVER_CHECK(cusolverDnZgetrf, M, N, cucplx(A), LDA, cucplx(Workspace.data()), ipiv, &info_u);
    info = info_u;
  }

  void getrs(char op, int N, int NRHS, double const *A, int LDA, int const *ipiv, double *B, int LDB, int &info) {
    CUSOLVER_CHECK(cusolverDnDgetrs, get_cublas_op(op), N, NRHS, A, LDA, ipiv, B, LDB, &info_u);
    info = info_u;
  }
  void getrs(char op, int N, int NRHS, std::complex<double> const *A, int LDA, int const *ipiv, std::complex<double> *B, int LDB, int &info) {
    CUSOLVER_CHECK(cusolverDnZgetrs, get_cublas_op(op), N, NRHS, cucplx(A), LDA, ipiv, cucplx(B), LDB, &info_u);
    info = info_u;
  }

} // namespace nda::lapack::cuda
