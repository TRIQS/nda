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
#include "../../traits.hpp"

#include <cusolverDn.h>

#include <string>
#include <type_traits>

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

  // Anonymous namespace for some file local helper functions.
  namespace {

    // Get the buffer size for gesvd.
    template <typename T>
    int gesvd_buffer_size_impl(int m, int n) {
      int bufferSize = 0;
      if constexpr (std::is_same_v<T, float>) {
        cusolverDnSgesvd_bufferSize(get_handle(), m, n, &bufferSize);
      } else if constexpr (std::is_same_v<T, double>) {
        cusolverDnDgesvd_bufferSize(get_handle(), m, n, &bufferSize);
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        cusolverDnCgesvd_bufferSize(get_handle(), m, n, &bufferSize);
      } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        cusolverDnZgesvd_bufferSize(get_handle(), m, n, &bufferSize);
      }
      return bufferSize;
    }

  } // namespace

  // gesvd buffer size
  int gesvd_buffer_size(int m, int n, float *) { return gesvd_buffer_size_impl<float>(m, n); }
  int gesvd_buffer_size(int m, int n, std::complex<float> *) { return gesvd_buffer_size_impl<std::complex<float>>(m, n); }
  int gesvd_buffer_size(int m, int n, double *) { return gesvd_buffer_size_impl<double>(m, n); }
  int gesvd_buffer_size(int m, int n, std::complex<double> *) { return gesvd_buffer_size_impl<std::complex<double>>(m, n); }

  // gesvd
  void gesvd(char jobu, char jobvt, int m, int n, float *a, int lda, float *s, float *u, int ldu, float *vt, int ldvt, float *work, int lwork,
             float *rwork, int &info) {
    CUSOLVER_CHECK(cusolverDnSgesvd, info, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork);
  }
  void gesvd(char jobu, char jobvt, int m, int n, std::complex<float> *a, int lda, float *s, std::complex<float> *u, int ldu, std::complex<float> *vt,
             int ldvt, std::complex<float> *work, int lwork, float *rwork, int &info) {
    CUSOLVER_CHECK(cusolverDnCgesvd, info, jobu, jobvt, m, n, cucplx(a), lda, s, cucplx(u), ldu, cucplx(vt), ldvt, cucplx(work), lwork, rwork);
  }
  void gesvd(char jobu, char jobvt, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt, double *work, int lwork,
             double *rwork, int &info) {
    CUSOLVER_CHECK(cusolverDnDgesvd, info, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork);
  }
  void gesvd(char jobu, char jobvt, int m, int n, std::complex<double> *a, int lda, double *s, std::complex<double> *u, int ldu,
             std::complex<double> *vt, int ldvt, std::complex<double> *work, int lwork, double *rwork, int &info) {
    CUSOLVER_CHECK(cusolverDnZgesvd, info, jobu, jobvt, m, n, cucplx(a), lda, s, cucplx(u), ldu, cucplx(vt), ldvt, cucplx(work), lwork, rwork);
  }

  void getrf(int m, int n, double *a, int lda, int *ipiv, int &info) {
    int bufferSize = 0;
    cusolverDnDgetrf_bufferSize(get_handle(), m, n, a, lda, &bufferSize);
    auto Workspace = nda::cuvector<double>(bufferSize);
    CUSOLVER_CHECK(cusolverDnDgetrf, info, m, n, a, lda, Workspace.data(), ipiv);
  }
  void getrf(int m, int n, std::complex<double> *a, int lda, int *ipiv, int &info) {
    int bufferSize = 0;
    cusolverDnZgetrf_bufferSize(get_handle(), m, n, cucplx(a), lda, &bufferSize);
    auto Workspace = nda::cuvector<std::complex<double>>(bufferSize);
    CUSOLVER_CHECK(cusolverDnZgetrf, info, m, n, cucplx(a), lda, cucplx(Workspace.data()), ipiv);
  }

  void getrs(char op, int n, int nrhs, double const *a, int lda, int const *ipiv, double *b, int ldb, int &info) {
    CUSOLVER_CHECK(cusolverDnDgetrs, info, get_cublas_op(op), n, nrhs, a, lda, ipiv, b, ldb);
  }
  void getrs(char op, int n, int nrhs, std::complex<double> const *a, int lda, int const *ipiv, std::complex<double> *b, int ldb, int &info) {
    CUSOLVER_CHECK(cusolverDnZgetrs, info, get_cublas_op(op), n, nrhs, cucplx(a), lda, ipiv, cucplx(b), ldb);
  }

} // namespace nda::lapack::device
