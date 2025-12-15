// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Implementation details for lapack/interface/cusolver_interface.hpp.
 */
#include <mpi/mpi.hpp>
#include <nda/nda.hpp>
#include "./cusolver_interface.hpp"
#include "../../basic_array.hpp"
#include "../../blas/tools.hpp"
#include "../../declarations.hpp"
#include "../../device.hpp"
#include "../../exceptions.hpp"
#include "../../macros.hpp"
#include "../../mem/allocators.hpp"
#include "../../mem/handle.hpp"
#include "../../mem/fill.hpp"
#include "../../mem/memcpy.hpp"
#include "cxx_interface.hpp"

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
  if (err != CUSOLVER_STATUS_SUCCESS) {                                                                                                              \
    std::cerr << AS_STRING(X) << " failed with error code " << std::to_string(err) << std::endl;                                                     \
    mpi::communicator{}.abort(11);                                                                                                                   \
  }                                                                                                                                                  \
  if (synchronize) {                                                                                                                                 \
    auto err1 = cudaDeviceSynchronize();                                                                                                             \
    if (err1 != cudaSuccess) {                                                                                                                       \
      std::cerr << " cudaDeviceSynchronize failed after call to: " << AS_STRING(X) " \n "                                                            \
                << " cudaGetErrorName: " << std::string(cudaGetErrorName(err1)) << "\n"                                                              \
                << " cudaGetErrorString: " << std::string(cudaGetErrorString(err1)) << std::endl;                                                    \
      mpi::communicator{}.abort(11);                                                                                                                 \
    }                                                                                                                                                \
  }                                                                                                                                                  \
  info = *get_info_ptr();

  void gesvd(char JOBU, char JOBVT, int M, int N, float *A, int LDA, float *S, float *U, int LDU, float *VT, int LDVT, float *WORK, int LWORK,
             float *RWORK, int &INFO) {
    // Replicate behavior of Netlib gesvd
    if (LWORK == -1) {
      int bufferSize = 0;
      cusolverDnSgesvd_bufferSize(get_handle(), M, N, &bufferSize);
      *WORK = bufferSize;
    } else {
      CUSOLVER_CHECK(cusolverDnSgesvd, INFO, JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, RWORK);
    }
  }
  void gesvd(char JOBU, char JOBVT, int M, int N, fcomplex *A, int LDA, float *S, fcomplex *U, int LDU, fcomplex *VT, int LDVT, fcomplex *WORK,
             int LWORK, float *RWORK, int &INFO) {
    // Replicate behavior of Netlib gesvd
    if (LWORK == -1) {
      int bufferSize = 0;
      cusolverDnCgesvd_bufferSize(get_handle(), M, N, &bufferSize);
      *WORK = bufferSize;
    } else {
      CUSOLVER_CHECK(cusolverDnCgesvd, INFO, JOBU, JOBVT, M, N, to_cublas(A), LDA, S, to_cublas(U), LDU, to_cublas(VT), LDVT, to_cublas(WORK), LWORK,
                     RWORK); // NOLINT
    }
  }
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
      CUSOLVER_CHECK(cusolverDnZgesvd, INFO, JOBU, JOBVT, M, N, to_cublas(A), LDA, S, to_cublas(U), LDU, to_cublas(VT), LDVT, to_cublas(WORK), LWORK,
                     RWORK); // NOLINT
    }
  }

// getrf
#define _getrf_bufferSize_(FUN, TYPE)                                                                                                                \
  int getrf_bufferSize(int M, int N, TYPE *A, int LDA) {                                                                                             \
    int bufferSize = 0;                                                                                                                              \
    FUN(get_handle(), M, N, to_cublas(A), LDA, &bufferSize);                                                                                         \
    return bufferSize;                                                                                                                               \
  }
  _getrf_bufferSize_(cusolverDnSgetrf_bufferSize, float) _getrf_bufferSize_(cusolverDnDgetrf_bufferSize, double)
     _getrf_bufferSize_(cusolverDnCgetrf_bufferSize, std::complex<float>) _getrf_bufferSize_(cusolverDnZgetrf_bufferSize, std::complex<double>)

#define _getrf_(FUN, TYPE)                                                                                                                           \
  void getrf(int M, int N, TYPE *A, int LDA, TYPE *W, int *ipiv, int &info) {                                                                        \
    CUSOLVER_CHECK(FUN, info, M, N, to_cublas(A), LDA, to_cublas(W), ipiv);                                                                          \
  }
        _getrf_(cusolverDnSgetrf, float) _getrf_(cusolverDnDgetrf, double) _getrf_(cusolverDnCgetrf, std::complex<float>)
           _getrf_(cusolverDnZgetrf, std::complex<double>)

// getrs
#define _getrs_(FUN, TYPE)                                                                                                                           \
  void getrs(char op, int N, int NRHS, TYPE const *A, int LDA, int const *ipiv, TYPE *B, int LDB, int &info) {                                       \
    CUSOLVER_CHECK(FUN, info, get_cublas_op(op), N, NRHS, to_cublas(A), LDA, ipiv, to_cublas(B), LDB);                                               \
  }
              _getrs_(cusolverDnSgetrs, float) _getrs_(cusolverDnDgetrs, double) _getrs_(cusolverDnCgetrs, std::complex<float>)
                 _getrs_(cusolverDnZgetrs, std::complex<double>)

// use getrs with B=Idensity
#define _getri_(TYPE)                                                                                                                                \
  void getri(int N, TYPE *A, int LDA, int const *ipiv, TYPE *WORK, int LWORK, int &info) {                                                           \
    using mem::Device;                                                                                                                               \
    if (LWORK >= N * N) {                                                                                                                            \
      auto B = nda::cuarray_view<TYPE, 2>(std::array<long, 2>{N, N}, WORK);                                                                          \
      B()    = TYPE(0.0);                                                                                                                            \
      mem::fill2D_n<mem::Device>(B.data(), N + 1, 1, N, TYPE(1.0));                                                                                  \
      getrs('N', N, N, A, LDA, ipiv, B.data(), N, info);                                                                                             \
      mem::memcpy2D<Device, Device>(A, LDA * sizeof(TYPE), WORK, N * sizeof(TYPE), N * sizeof(TYPE), N);                                             \
    } else {                                                                                                                                         \
      auto B = nda::cuvector<TYPE>(N * N);                                                                                                           \
      B()    = TYPE(0.0);                                                                                                                            \
      mem::fill2D_n<mem::Device>(B.data(), N + 1, 1, N, TYPE(1.0));                                                                                  \
      getrs('N', N, N, A, LDA, ipiv, B.data(), N, info);                                                                                             \
      mem::memcpy2D<Device, Device>(A, LDA * sizeof(TYPE), WORK, N * sizeof(TYPE), N * sizeof(TYPE), N);                                             \
    }                                                                                                                                                \
  }
                    _getri_(float) _getri_(double) _getri_(std::complex<float>) _getri_(std::complex<double>)

// geqrf
#define _geqrf_bufferSize_(FUN, TYPE)                                                                                                                \
  int geqrf_bufferSize(int M, int N, TYPE *A, int LDA) {                                                                                             \
    int bufferSize = 0;                                                                                                                              \
    FUN(get_handle(), M, N, to_cublas(A), LDA, &bufferSize);                                                                                         \
    return bufferSize;                                                                                                                               \
  }
                       _geqrf_bufferSize_(cusolverDnSgeqrf_bufferSize, float) _geqrf_bufferSize_(cusolverDnDgeqrf_bufferSize, double)
                          _geqrf_bufferSize_(cusolverDnCgeqrf_bufferSize, std::complex<float>)
                             _geqrf_bufferSize_(cusolverDnZgeqrf_bufferSize, std::complex<double>)

#define _geqrf_(FUN, TYPE)                                                                                                                           \
  void geqrf(int M, int N, TYPE *A, int LDA, TYPE *tau, TYPE *W, int Lwork, int &info) {                                                             \
    CUSOLVER_CHECK(FUN, info, M, N, to_cublas(A), LDA, to_cublas(tau), to_cublas(W), Lwork);                                                         \
  }
                                _geqrf_(cusolverDnSgeqrf, float) _geqrf_(cusolverDnDgeqrf, double) _geqrf_(cusolverDnCgeqrf, std::complex<float>)
                                   _geqrf_(cusolverDnZgeqrf, std::complex<double>)

// orgqr
#define _orgqr_bufferSize_(FUN, TYPE)                                                                                                                \
  int orgqr_bufferSize(int M, int N, int K, const TYPE *A, int LDA, const TYPE *tau) {                                                               \
    int bufferSize = 0;                                                                                                                              \
    FUN(get_handle(), M, N, K, to_cublas(A), LDA, to_cublas(tau), &bufferSize);                                                                      \
    return bufferSize;                                                                                                                               \
  }
                                      _orgqr_bufferSize_(cusolverDnSorgqr_bufferSize, float) _orgqr_bufferSize_(cusolverDnDorgqr_bufferSize, double)

// ungqr
#define _ungqr_bufferSize_(FUN, TYPE)                                                                                                                \
  int ungqr_bufferSize(int M, int N, int K, const TYPE *A, int LDA, const TYPE *tau) {                                                               \
    int bufferSize = 0;                                                                                                                              \
    FUN(get_handle(), M, N, K, to_cublas(A), LDA, to_cublas(tau), &bufferSize);                                                                      \
    return bufferSize;                                                                                                                               \
  }
                                         _ungqr_bufferSize_(cusolverDnCungqr_bufferSize, std::complex<float>)
                                            _ungqr_bufferSize_(cusolverDnZungqr_bufferSize, std::complex<double>)

#define _orgqr_(FUN, TYPE)                                                                                                                           \
  void orgqr(int M, int N, int K, TYPE *A, int LDA, TYPE const *tau, TYPE *W, int Lwork, int &info) {                                                \
    CUSOLVER_CHECK(FUN, info, M, N, K, to_cublas(A), LDA, to_cublas(tau), to_cublas(W), Lwork);                                                      \
  }
                                               _orgqr_(cusolverDnSorgqr, float) _orgqr_(cusolverDnDorgqr, double)

#define _ungqr_(FUN, TYPE)                                                                                                                           \
  void ungqr(int M, int N, int K, TYPE *A, int LDA, TYPE const *tau, TYPE *W, int Lwork, int &info) {                                                \
    CUSOLVER_CHECK(FUN, info, M, N, K, to_cublas(A), LDA, to_cublas(tau), to_cublas(W), Lwork);                                                      \
  }
                                                  _ungqr_(cusolverDnCungqr, std::complex<float>) _ungqr_(cusolverDnZungqr, std::complex<double>)

} // namespace nda::lapack::device
