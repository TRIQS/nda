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
// Authors: Miguel Morales, Nils Wentzell, Geraud Krawezik

#include <nda/nda.hpp>
#include <nda/macros.hpp>
#include <nda/exceptions.hpp>
#include "cxx_interface.hpp"

#include "hip/hip_runtime_api.h"
#include "rocsolver/rocsolver.h"

#include <string>

using namespace std::string_literals;

namespace nda::lapack::device {

  // Local function to get unique rocblas_handle Handle, Used by all routines
  // NOTE: using rocblas_handle as the documentation mentions rocm_solver is deprecated in rocsolver/rocsolver-aliases.h
  inline rocblas_handle &get_handle() {
    struct handle_storage_t { // RAII for handle
      handle_storage_t() { rocblas_create_handle(&handle); }
      ~handle_storage_t() { rocblas_destroy_handle(handle); }
      rocblas_handle handle = {};
    };
    static auto sto = handle_storage_t{};
    return sto.handle;
  }

  // Get Integer Pointer in unified memory, Used to return info from lapack routines
  int *get_info_ptr() {
    static auto info_u_handle = mem::handle_heap<int, mem::mallocator<mem::Unified>>(1);
    return info_u_handle.data();
  }

  /// Global option to turn on/off the hipDeviceSynchronize after rocsolver library calls
  static bool synchronize = true;
#define ROCSOLVER_CHECK(X, info, ...)                                                                                                                \
  auto err = X(get_handle(), __VA_ARGS__, get_info_ptr());                                                                                           \
  if (err != rocblas_status_success) { NDA_RUNTIME_ERROR << AS_STRING(X) << " failed with error code " << std::to_string(err); }                     \
  if (synchronize) {                                                                                                                                 \
    auto errsync = hipDeviceSynchronize();                                                                                                           \
    if (errsync != hipSuccess) {                                                                                                                     \
      NDA_RUNTIME_ERROR << " hipDeviceSynchronize failed after call to: " << AS_STRING(X) " \n "                                                     \
                        << " hipGetErrorName: " << std::string(hipGetErrorName(errsync)) << "\n"                                                     \
                        << " hipGetErrorString: " << std::string(hipGetErrorString(errsync)) << "\n";                                                \
    }                                                                                                                                                \
  }                                                                                                                                                  \
  info = *get_info_ptr();

#define ROCSOLVER_CHECK_INFO_END(X, info, ...)                                                                                                       \
  auto err = X(get_handle(), __VA_ARGS__, get_info_ptr());                                                                                           \
  if (err != rocblas_status_success) { NDA_RUNTIME_ERROR << AS_STRING(X) << " failed with error code " << std::to_string(err); }                     \
  if (synchronize) {                                                                                                                                 \
    auto errsync = hipDeviceSynchronize();                                                                                                           \
    if (errsync != hipSuccess) {                                                                                                                     \
      NDA_RUNTIME_ERROR << " hipDeviceSynchronize failed after call to: " << AS_STRING(X) " \n "                                                     \
                        << " hipGetErrorName: " << std::string(hipGetErrorName(errsync)) << "\n"                                                     \
                        << " hipGetErrorString: " << std::string(hipGetErrorString(errsync)) << "\n";                                                \
    }                                                                                                                                                \
  }                                                                                                                                                  \
  info = *get_info_ptr();

#define ROCSOLVER_CHECK_INFO_NONE(X, info, ...)                                                                                                      \
  auto err = X(get_handle(), __VA_ARGS__);                                                                                                           \
  if (err != rocblas_status_success) { NDA_RUNTIME_ERROR << AS_STRING(X) << " failed with error code " << std::to_string(err); }                     \
  if (synchronize) {                                                                                                                                 \
    auto errsync = hipDeviceSynchronize();                                                                                                           \
    if (errsync != hipSuccess) {                                                                                                                     \
      NDA_RUNTIME_ERROR << " hipDeviceSynchronize failed after call to: " << AS_STRING(X) " \n "                                                     \
                        << " hipGetErrorName: " << std::string(hipGetErrorName(errsync)) << "\n"                                                     \
                        << " hipGetErrorString: " << std::string(hipGetErrorString(errsync)) << "\n";                                                \
    }                                                                                                                                                \
  }                                                                                                                                                  \
  //info = nullptr;

  void gesvd(char JOBU, char JOBVT, int M, int N, double *A, int LDA, double *S, double *U, int LDU, double *VT, int LDVT, double *WORK, int LWORK,
             double *RWORK, int &INFO) {
    if (LWORK == -1)
      NDA_RUNTIME_ERROR << "gesvd with LWORK == -1 is not supported on ROCm devices\n";
    else {
      ROCSOLVER_CHECK_INFO_END(rocsolver_dgesvd, INFO, get_rocblas_svect(JOBU), get_rocblas_svect(JOBVT), M, N, A, LDA, S, U, LDU, VT, LDVT, RWORK,
                               rocblas_inplace);
    }
  }
  void gesvd(char JOBU, char JOBVT, int M, int N, dcomplex *A, int LDA, double *S, dcomplex *U, int LDU, dcomplex *VT, int LDVT, dcomplex *WORK,
             int LWORK, double *RWORK, int &INFO) {
    if (LWORK == -1)
      NDA_RUNTIME_ERROR << "gesvd with LWORK == -1 is not supported on ROCm devices\n";
    else {
      ROCSOLVER_CHECK_INFO_END(rocsolver_zgesvd, INFO, get_rocblas_svect(JOBU), get_rocblas_svect(JOBVT), M, N, roccplx(A), LDA, S, roccplx(U), LDU,
                               roccplx(VT), LDVT, RWORK, rocblas_inplace); // NOLINT
    }
  }

  void getrf(int M, int N, double *A, int LDA, int *ipiv, int &info) { ROCSOLVER_CHECK_INFO_END(rocsolver_dgetrf, info, M, N, A, LDA, ipiv); }
  void getrf(int M, int N, dcomplex *A, int LDA, int *ipiv, int &info) {
    ROCSOLVER_CHECK_INFO_END(rocsolver_zgetrf, info, M, N, roccplx(A), LDA, ipiv);
  }

  void getrs(char op, int N, int NRHS, double const *A, int LDA, int const *ipiv, double *B, int LDB, int &info) {
    ROCSOLVER_CHECK_INFO_NONE(rocsolver_dgetrs, info, get_rocblas_op(op), N, NRHS, const_cast<double *>(A), LDA, ipiv, B, LDB);
  }
  void getrs(char op, int N, int NRHS, dcomplex const *A, int LDA, int const *ipiv, dcomplex *B, int LDB, int &info) {
    ROCSOLVER_CHECK_INFO_NONE(rocsolver_zgetrs, info, get_rocblas_op(op), N, NRHS, roccplx(const_cast<std::complex<double> *>(A)), LDA, ipiv,
                              roccplx(B), LDB);
  }

} // namespace nda::lapack::device
