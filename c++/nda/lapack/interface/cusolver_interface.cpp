// Copyright (c) 2022-2023 Simons Foundation
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
// Authors: Miguel Morales, Nils Wentzell

#include <nda/nda.hpp>
#include <nda/macros.hpp>
#include <nda/exceptions.hpp>
#include "cxx_interface.hpp"

#include "cuda_runtime.h"
#include "cusolverDn.h"

#include <string>

using namespace std::string_literals;

namespace nda::lapack::device {

  // Local function to get unique CuSolver Handle, Used by all routines
  inline cusolverDnHandle_t &get_handle() {
    struct handle_storage_t { // RAII for handle
      handle_storage_t() { cusolverDnCreate(&handle); }
      ~handle_storage_t() { cusolverDnDestroy(handle); }
      cusolverDnHandle_t handle = {};
    };
    static auto sto = handle_storage_t{};
    return sto.handle;
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
  if (err != CUSOLVER_STATUS_SUCCESS) { NDA_RUNTIME_ERROR << AS_STRING(X) << " failed with error code " << std::to_string(err); }                    \
  if (synchronize) {                                                                                                                                 \
    auto errsync = cudaDeviceSynchronize();                                                                                                          \
    if (errsync != cudaSuccess) {                                                                                                                    \
      NDA_RUNTIME_ERROR << " cudaDeviceSynchronize failed after call to: " << AS_STRING(X) " \n "                                                    \
                        << " cudaGetErrorName: " << std::string(cudaGetErrorName(errsync)) << "\n"                                                   \
                        << " cudaGetErrorString: " << std::string(cudaGetErrorString(errsync)) << "\n";                                              \
    }                                                                                                                                                \
  }                                                                                                                                                  \
  info = *get_info_ptr();

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
  void gesvd(char JOBU, char JOBVT, int M, int N, dcomplex *A, int LDA, double *S, dcomplex *U, int LDU, dcomplex *VT, int LDVT, dcomplex *WORK,
             int LWORK, double *RWORK, int &INFO) {
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
  void getrf(int M, int N, dcomplex *A, int LDA, int *ipiv, int &info) {
    int bufferSize = 0;
    cusolverDnZgetrf_bufferSize(get_handle(), M, N, cucplx(A), LDA, &bufferSize);
    auto Workspace = nda::cuvector<dcomplex>(bufferSize);
    CUSOLVER_CHECK(cusolverDnZgetrf, info, M, N, cucplx(A), LDA, cucplx(Workspace.data()), ipiv);
  }

  void getrs(char op, int N, int NRHS, double const *A, int LDA, int const *ipiv, double *B, int LDB, int &info) {
    CUSOLVER_CHECK(cusolverDnDgetrs, info, get_cublas_op(op), N, NRHS, A, LDA, ipiv, B, LDB);
  }
  void getrs(char op, int N, int NRHS, dcomplex const *A, int LDA, int const *ipiv, dcomplex *B, int LDB, int &info) {
    CUSOLVER_CHECK(cusolverDnZgetrs, info, get_cublas_op(op), N, NRHS, cucplx(A), LDA, ipiv, cucplx(B), LDB);
  }

} // namespace nda::lapack::device
