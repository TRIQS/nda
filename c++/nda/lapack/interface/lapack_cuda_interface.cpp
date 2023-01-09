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
#include <nda/mem/fill.hpp>
#include "lapack_cxx_interface.hpp"

#include "cusolverDn.h"

#include <string>

using namespace std::string_literals;

namespace nda::lapack::device {

  constexpr cublasOperation_t get_cublas_op(char op) {
    switch (op) {
      case 'N': return CUBLAS_OP_N; break;
      case 'T': return CUBLAS_OP_T; break;
      case 'C': return CUBLAS_OP_C; break;
      default: std::terminate(); return {};
    }
  }

  // Get CuSolver Handle, Used by all routines
  auto &get_handle() {
    struct handle_t {
      handle_t() { cusolverDnCreate(&h); }
      ~handle_t() { cusolverDnDestroy(h); }
      operator cusolverDnHandle_t() { return h; }

      private:
      cusolverDnHandle_t h = {};
    };
    static auto h = handle_t{};
    return h;
  }

  // Get Integer Pointer in unified memory, Used to return info from lapack routines
  int *get_info_ptr() {
    static auto info_u_handle = mem::handle_heap<int, mem::mallocator<mem::Unified>>(1);
    return info_u_handle.data();
  }

  /// Global option to turn on/off the cudaDeviceSynchronize after cusolver library calls
  static bool synchronize = true;
#define CUSOLVER_CHECK(X, info, ...)                                                                                                                 \
  auto err = X(get_handle(), __VA_ARGS__, get_info_ptr());                                                                                           \
  info     = *get_info_ptr();                                                                                                                        \
  if (synchronize) cudaDeviceSynchronize();                                                                                                          \
  if (err != CUSOLVER_STATUS_SUCCESS) NDA_RUNTIME_ERROR << AS_STRING(X) + " failed with error code "s + std::to_string(err);

  inline auto *cucplx(std::complex<double> *c) { return reinterpret_cast<cuDoubleComplex *>(c); }             // NOLINT
  inline auto *cucplx(std::complex<double> const *c) { return reinterpret_cast<const cuDoubleComplex *>(c); } // NOLINT

  void gesvd(char JOBU, char JOBVT, int M, int N, double *A, int LDA, double *S, double *U, int LDU, double *VT, int LDVT, double *WORK, int LWORK,
             double *RWORK, int &INFO) {
    // Replicate behavior of Netlib gesvd
    if (LWORK == -1) {
      int bufferSize = 0;
      cusolverDnDgesvd_bufferSize(get_handle(), M, N, &bufferSize);
      *WORK = bufferSize;
    } else {
      CUSOLVER_CHECK(cusolverDnDgesvd, INFO, JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, RWORK);
    }
  }
  void gesvd(char JOBU, char JOBVT, int M, int N, std::complex<double> *A, int LDA, double *S, std::complex<double> *U, int LDU,
             std::complex<double> *VT, int LDVT, std::complex<double> *WORK, int LWORK, double *RWORK, int &INFO) {
    // Replicate behavior of Netlib gesvd
    if (LWORK == -1) {
      int bufferSize = 0;
      cusolverDnZgesvd_bufferSize(get_handle(), M, N, &bufferSize);
      *WORK = bufferSize;
    } else {
      CUSOLVER_CHECK(cusolverDnZgesvd, INFO, JOBU, JOBVT, M, N, cucplx(A), LDA, S, cucplx(U), LDU, cucplx(VT), LDVT, cucplx(WORK), LWORK,
                     RWORK); // NOLINT
    }
  }

  void getrf(int M, int N, double *A, int LDA, int *ipiv, int &info) {
    int bufferSize = 0;
    cusolverDnDgetrf_bufferSize(get_handle(), M, N, A, LDA, &bufferSize);
    auto Workspace = nda::cuvector<double>(bufferSize);
    CUSOLVER_CHECK(cusolverDnDgetrf, info, M, N, A, LDA, Workspace.data(), ipiv);
  }
  void getrf(int M, int N, std::complex<double> *A, int LDA, int *ipiv, int &info) {
    int bufferSize = 0;
    cusolverDnZgetrf_bufferSize(get_handle(), M, N, cucplx(A), LDA, &bufferSize);
    auto Workspace = nda::cuvector<dcomplex>(bufferSize);
    CUSOLVER_CHECK(cusolverDnZgetrf, info, M, N, cucplx(A), LDA, cucplx(Workspace.data()), ipiv);
  }

  void getrs(char op, int N, int NRHS, double const *A, int LDA, int const *ipiv, double *B, int LDB, int &info) {
    CUSOLVER_CHECK(cusolverDnDgetrs, info, get_cublas_op(op), N, NRHS, A, LDA, ipiv, B, LDB);
  }
  void getrs(char op, int N, int NRHS, std::complex<double> const *A, int LDA, int const *ipiv, std::complex<double> *B, int LDB, int &info) {
    CUSOLVER_CHECK(cusolverDnZgetrs, info, get_cublas_op(op), N, NRHS, cucplx(A), LDA, ipiv, cucplx(B), LDB);
  }

  // use getrs with B=Idensity
  void getri(int N, double *A, int LDA, int *ipiv, double *WORK, int LWORK, int &info)
  {
    if (LWORK == -1) {
      int bufferSize = N*N;
      *WORK = bufferSize;
      return;
    }
    if(LWORK == N*N) {
      auto B = nda::cuarray_view<double,2>(std::array<long,2>{N,N},WORK);
      B() = 0.0;
      mem::fill2D_n<mem::Device>(B.data(), N+1, 1, N, 1.0);
      getrs('N',N,N,A,LDA,ipiv,B.data(),N,info);
    } else {
      auto B = nda::cuvector<double>(N*N);
      B() = 0.0;
      mem::fill2D_n<mem::Device>(B.data(), N+1, 1, N, 1.0);
      getrs('N',N,N,A,LDA,ipiv,B.data(),N,info);
    }
  }
  void getri(int N, std::complex<double> *A, int LDA, int *ipiv, std::complex<double> *WORK, int LWORK, int &info)
  {
    if (LWORK == -1) {
      int bufferSize = N*N;
      *WORK = bufferSize;
      return;
    }
    if(LWORK == N*N) {
      auto B = nda::cuarray_view<dcomplex,2>(std::array<long,2>{N,N},WORK);
      B() = 0.0;
      mem::fill2D_n<mem::Device>(B.data(), N+1, 1, N, 1.0);
      getrs('N',N,N,cucplx(A),LDA,ipiv,cucplx(B.data()),N,info);
    } else {
      auto B = nda::cuvector<dcomplex>(N*N);
      B() = 0.0;
      mem::fill2D_n<mem::Device>(B.data(), N+1, 1, N, 1.0);
      getrs('N',N,N,cucplx(A),LDA,ipiv,cucplx(B.data()),N,info);
    }
  }

} // namespace nda::lapack::device
