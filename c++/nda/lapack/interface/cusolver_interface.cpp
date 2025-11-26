// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Implementation details for lapack/interface/cusolver_interface.hpp.
 */

#include "./cusolver_interface.hpp"
#include "../../basic_array.hpp"
#include "../../blas/tools.hpp"
#include "../../declarations.hpp"
#include "../../device.hpp"
#include "../../exceptions.hpp"
#include "../../macros.hpp"
#include "../../mem/allocators.hpp"
#include "../../mem/handle.hpp"

#include <cusolverDn.h>

#include <string>

namespace nda::lapack::device {

  // Local function to get unique CuSolver handle.
  inline cusolverDnHandle_t &get_handle() {
    struct handle_storage_t { // RAII for handle
      handle_storage_t() { cusolverDnCreate(&handle); }
      ~handle_storage_t() { cusolverDnDestroy(handle); }
      cusolverDnHandle_t handle = {};
    };
    static auto sto = handle_storage_t{};
    return sto.handle;
  }

  // Get an integer pointer in unified memory to return info from lapack routines.
  int *get_info_ptr() {
    static auto info_u_handle = mem::handle_heap<int, mem::mallocator<mem::Unified>>(1);
    return info_u_handle.data();
  }

  // Global option to turn on/off the cudaDeviceSynchronize after cusolver library calls.
  static bool synchronize = true; // NOLINT  (global option is on purpose)

// Macro to check cusolver calls.
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

  void gesvd(char jobu, char jobvt, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt, double *work, int lwork,
             double *rwork, int &info) {
    // Replicate behavior of Netlib gesvd
    if (lwork == -1) {
      int bufferSize = 0;
      cusolverDnDgesvd_bufferSize(get_handle(), m, n, &bufferSize);
      *work = bufferSize;
    } else {
      CUSOLVER_CHECK(cusolverDnDgesvd, info, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork);
    }
  }
  void gesvd(char jobu, char jobvt, int m, int n, dcomplex *a, int lda, double *s, dcomplex *u, int ldu, dcomplex *vt, int ldvt, dcomplex *work,
             int lwork, double *rwork, int &info) {
    // Replicate behavior of Netlib gesvd
    if (lwork == -1) {
      int bufferSize = 0;
      cusolverDnZgesvd_bufferSize(get_handle(), m, n, &bufferSize);
      *work = bufferSize;
    } else {
      CUSOLVER_CHECK(cusolverDnZgesvd, info, jobu, jobvt, m, n, cucplx(a), lda, s, cucplx(u), ldu, cucplx(vt), ldvt, cucplx(work), lwork,
                     rwork); // NOLINT
    }
  }

  void getrf(int m, int n, double *a, int lda, int *ipiv, int &info) {
    int bufferSize = 0;
    cusolverDnDgetrf_bufferSize(get_handle(), m, n, a, lda, &bufferSize);
    auto Workspace = nda::cuvector<double>(bufferSize);
    CUSOLVER_CHECK(cusolverDnDgetrf, info, m, n, a, lda, Workspace.data(), ipiv);
  }
  void getrf(int m, int n, dcomplex *a, int lda, int *ipiv, int &info) {
    int bufferSize = 0;
    cusolverDnZgetrf_bufferSize(get_handle(), m, n, cucplx(a), lda, &bufferSize);
    auto Workspace = nda::cuvector<dcomplex>(bufferSize);
    CUSOLVER_CHECK(cusolverDnZgetrf, info, m, n, cucplx(a), lda, cucplx(Workspace.data()), ipiv);
  }

  void getrs(char op, int n, int nrhs, double const *a, int lda, int const *ipiv, double *b, int ldb, int &info) {
    CUSOLVER_CHECK(cusolverDnDgetrs, info, get_cublas_op(op), n, nrhs, a, lda, ipiv, b, ldb);
  }
  void getrs(char op, int n, int nrhs, dcomplex const *a, int lda, int const *ipiv, dcomplex *b, int ldb, int &info) {
    CUSOLVER_CHECK(cusolverDnZgetrs, info, get_cublas_op(op), n, nrhs, cucplx(a), lda, ipiv, cucplx(b), ldb);
  }

} // namespace nda::lapack::device
