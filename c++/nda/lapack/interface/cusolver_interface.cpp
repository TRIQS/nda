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
#include "../../mem/fill.hpp"
#include "../../mem/handle.hpp"
#include "../../mem/memcpy.hpp"
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

    // Get the buffer size for getrf.
    template <typename T>
    int getrf_buffer_size_impl(int m, int n, T *a, int lda) {
      int bufferSize = 0;
      if constexpr (std::is_same_v<T, float>) {
        cusolverDnSgetrf_bufferSize(get_handle(), m, n, a, lda, &bufferSize);
      } else if constexpr (std::is_same_v<T, double>) {
        cusolverDnDgetrf_bufferSize(get_handle(), m, n, a, lda, &bufferSize);
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        cusolverDnCgetrf_bufferSize(get_handle(), m, n, cucplx(a), lda, &bufferSize);
      } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        cusolverDnZgetrf_bufferSize(get_handle(), m, n, cucplx(a), lda, &bufferSize);
      }
      return bufferSize;
    }

    // Get the buffer size for geqrf.
    template <typename T>
    int geqrf_buffer_size_impl(int m, int n, T *a, int lda) {
      int bufferSize = 0;
      if constexpr (std::is_same_v<T, float>) {
        cusolverDnSgeqrf_bufferSize(get_handle(), m, n, a, lda, &bufferSize);
      } else if constexpr (std::is_same_v<T, double>) {
        cusolverDnDgeqrf_bufferSize(get_handle(), m, n, a, lda, &bufferSize);
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        cusolverDnCgeqrf_bufferSize(get_handle(), m, n, cucplx(a), lda, &bufferSize);
      } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        cusolverDnZgeqrf_bufferSize(get_handle(), m, n, cucplx(a), lda, &bufferSize);
      }
      return bufferSize;
    }

    // Get the buffer size for orgqr/ungqr.
    template <typename T>
    int xxgqr_buffer_size_impl(int m, int n, int k, T const *a, int lda, T const *tau) {
      int bufferSize = 0;
      if constexpr (std::is_same_v<T, float>) {
        cusolverDnSorgqr_bufferSize(get_handle(), m, n, k, a, lda, tau, &bufferSize);
      } else if constexpr (std::is_same_v<T, double>) {
        cusolverDnDorgqr_bufferSize(get_handle(), m, n, k, a, lda, tau, &bufferSize);
      } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        cusolverDnCungqr_bufferSize(get_handle(), m, n, k, cucplx(a), lda, cucplx(tau), &bufferSize);
      } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        cusolverDnZungqr_bufferSize(get_handle(), m, n, k, cucplx(a), lda, cucplx(tau), &bufferSize);
      }
      return bufferSize;
    }

    // Custom getri implementation: build I in workspace (or a fallback buffer), call getrs to solve A * X = I, then
    // copy the resulting inverse back over A.
    template <typename T>
    void getri_impl(int n, T *a, int lda, int const *ipiv, T *work, int lwork, int &info) {
      auto solve_and_copy_back = [&](T *b_ptr) {
        auto B = nda::cuarray_view<T, 2>{std::array<long, 2>{n, n}, b_ptr};
        B()    = T(0);
        mem::fill2D_n<mem::Device>(b_ptr, static_cast<size_t>(n) + 1, 1, n, T(1));
        getrs('N', n, n, a, lda, ipiv, b_ptr, n, info);
        mem::memcpy2D<mem::Device, mem::Device>(a, lda * sizeof(T), b_ptr, n * sizeof(T), n * sizeof(T), n);
      };
      if (lwork >= n * n) {
        solve_and_copy_back(work);
      } else {
        auto tmp = nda::cuvector<T>(n * n);
        solve_and_copy_back(tmp.data());
      }
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

  // getrf buffer size
  int getrf_buffer_size(int m, int n, float *a, int lda) { return getrf_buffer_size_impl<float>(m, n, a, lda); }
  int getrf_buffer_size(int m, int n, std::complex<float> *a, int lda) { return getrf_buffer_size_impl<std::complex<float>>(m, n, a, lda); }
  int getrf_buffer_size(int m, int n, double *a, int lda) { return getrf_buffer_size_impl<double>(m, n, a, lda); }
  int getrf_buffer_size(int m, int n, std::complex<double> *a, int lda) { return getrf_buffer_size_impl<std::complex<double>>(m, n, a, lda); }

  // getrf
  void getrf(int m, int n, float *a, int lda, float *work, int *ipiv, int &info) { CUSOLVER_CHECK(cusolverDnSgetrf, info, m, n, a, lda, work, ipiv); }
  void getrf(int m, int n, std::complex<float> *a, int lda, std::complex<float> *work, int *ipiv, int &info) {
    CUSOLVER_CHECK(cusolverDnCgetrf, info, m, n, cucplx(a), lda, cucplx(work), ipiv);
  }
  void getrf(int m, int n, double *a, int lda, double *work, int *ipiv, int &info) {
    CUSOLVER_CHECK(cusolverDnDgetrf, info, m, n, a, lda, work, ipiv);
  }
  void getrf(int m, int n, std::complex<double> *a, int lda, std::complex<double> *work, int *ipiv, int &info) {
    CUSOLVER_CHECK(cusolverDnZgetrf, info, m, n, cucplx(a), lda, cucplx(work), ipiv);
  }

  // getrs
  void getrs(char op, int n, int nrhs, float const *a, int lda, int const *ipiv, float *b, int ldb, int &info) {
    CUSOLVER_CHECK(cusolverDnSgetrs, info, get_cublas_op(op), n, nrhs, a, lda, ipiv, b, ldb);
  }
  void getrs(char op, int n, int nrhs, std::complex<float> const *a, int lda, int const *ipiv, std::complex<float> *b, int ldb, int &info) {
    CUSOLVER_CHECK(cusolverDnCgetrs, info, get_cublas_op(op), n, nrhs, cucplx(a), lda, ipiv, cucplx(b), ldb);
  }
  void getrs(char op, int n, int nrhs, double const *a, int lda, int const *ipiv, double *b, int ldb, int &info) {
    CUSOLVER_CHECK(cusolverDnDgetrs, info, get_cublas_op(op), n, nrhs, a, lda, ipiv, b, ldb);
  }
  void getrs(char op, int n, int nrhs, std::complex<double> const *a, int lda, int const *ipiv, std::complex<double> *b, int ldb, int &info) {
    CUSOLVER_CHECK(cusolverDnZgetrs, info, get_cublas_op(op), n, nrhs, cucplx(a), lda, ipiv, cucplx(b), ldb);
  }

  // getri
  void getri(int n, float *a, int lda, int const *ipiv, float *work, int lwork, int &info) { getri_impl<float>(n, a, lda, ipiv, work, lwork, info); }
  void getri(int n, std::complex<float> *a, int lda, int const *ipiv, std::complex<float> *work, int lwork, int &info) {
    getri_impl<std::complex<float>>(n, a, lda, ipiv, work, lwork, info);
  }
  void getri(int n, double *a, int lda, int const *ipiv, double *work, int lwork, int &info) {
    getri_impl<double>(n, a, lda, ipiv, work, lwork, info);
  }
  void getri(int n, std::complex<double> *a, int lda, int const *ipiv, std::complex<double> *work, int lwork, int &info) {
    getri_impl<std::complex<double>>(n, a, lda, ipiv, work, lwork, info);
  }

  // geqrf buffer size
  int geqrf_buffer_size(int m, int n, float *a, int lda) { return geqrf_buffer_size_impl<float>(m, n, a, lda); }
  int geqrf_buffer_size(int m, int n, std::complex<float> *a, int lda) { return geqrf_buffer_size_impl<std::complex<float>>(m, n, a, lda); }
  int geqrf_buffer_size(int m, int n, double *a, int lda) { return geqrf_buffer_size_impl<double>(m, n, a, lda); }
  int geqrf_buffer_size(int m, int n, std::complex<double> *a, int lda) { return geqrf_buffer_size_impl<std::complex<double>>(m, n, a, lda); }

  // geqrf
  void geqrf(int m, int n, float *a, int lda, float *tau, float *work, int lwork, int &info) {
    CUSOLVER_CHECK(cusolverDnSgeqrf, info, m, n, a, lda, tau, work, lwork);
  }
  void geqrf(int m, int n, std::complex<float> *a, int lda, std::complex<float> *tau, std::complex<float> *work, int lwork, int &info) {
    CUSOLVER_CHECK(cusolverDnCgeqrf, info, m, n, cucplx(a), lda, cucplx(tau), cucplx(work), lwork);
  }
  void geqrf(int m, int n, double *a, int lda, double *tau, double *work, int lwork, int &info) {
    CUSOLVER_CHECK(cusolverDnDgeqrf, info, m, n, a, lda, tau, work, lwork);
  }
  void geqrf(int m, int n, std::complex<double> *a, int lda, std::complex<double> *tau, std::complex<double> *work, int lwork, int &info) {
    CUSOLVER_CHECK(cusolverDnZgeqrf, info, m, n, cucplx(a), lda, cucplx(tau), cucplx(work), lwork);
  }

  // orgqr buffer size
  int orgqr_buffer_size(int m, int n, int k, float const *a, int lda, float const *tau) { return xxgqr_buffer_size_impl(m, n, k, a, lda, tau); }
  int orgqr_buffer_size(int m, int n, int k, double const *a, int lda, double const *tau) { return xxgqr_buffer_size_impl(m, n, k, a, lda, tau); }

  // orgqr
  void orgqr(int m, int n, int k, float *a, int lda, float const *tau, float *work, int lwork, int &info) {
    CUSOLVER_CHECK(cusolverDnSorgqr, info, m, n, k, a, lda, tau, work, lwork);
  }
  void orgqr(int m, int n, int k, double *a, int lda, double const *tau, double *work, int lwork, int &info) {
    CUSOLVER_CHECK(cusolverDnDorgqr, info, m, n, k, a, lda, tau, work, lwork);
  }

  // ungqr buffer size
  int ungqr_buffer_size(int m, int n, int k, std::complex<float> const *a, int lda, std::complex<float> const *tau) {
    return xxgqr_buffer_size_impl(m, n, k, a, lda, tau);
  }
  int ungqr_buffer_size(int m, int n, int k, std::complex<double> const *a, int lda, std::complex<double> const *tau) {
    return xxgqr_buffer_size_impl(m, n, k, a, lda, tau);
  }

  // ungqr
  void ungqr(int m, int n, int k, std::complex<float> *a, int lda, std::complex<float> const *tau, std::complex<float> *work, int lwork, int &info) {
    CUSOLVER_CHECK(cusolverDnCungqr, info, m, n, k, cucplx(a), lda, cucplx(tau), cucplx(work), lwork);
  }
  void ungqr(int m, int n, int k, std::complex<double> *a, int lda, std::complex<double> const *tau, std::complex<double> *work, int lwork,
             int &info) {
    CUSOLVER_CHECK(cusolverDnZungqr, info, m, n, k, cucplx(a), lda, cucplx(tau), cucplx(work), lwork);
  }

} // namespace nda::lapack::device
