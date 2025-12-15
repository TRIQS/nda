// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides GPU and non-GPU specific functionality.
 */

#pragma once

#ifdef NDA_HAVE_MPI
#include <mpi/mpi.hpp>
#endif

#ifdef NDA_HAVE_CUDA
#include "./exceptions.hpp"

#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <complex>
#include <iostream>
#include <exception>
#include <string>
#endif // NDA_HAVE_CUDA

namespace nda {

#ifdef NDA_HAVE_CUDA
  using devStream_t = cudaStream_t;
#else
  using devStream_t = int;
#endif

  /**
   * @addtogroup mem_utils
   * @{
   */

  /**
   * @brief Trigger a compilation error in case GPU specific functionality is used without configuring the project with
   * GPU support.
   */
  template <bool flag = false>
  void compile_error_no_gpu() {
    static_assert(flag, "Using device functionality without gpu support! Configure project with -DCudaSupport=ON.");
  }

#ifdef NDA_HAVE_CUDA

  /// Constexpr variable that is true if the project is configured with GPU support.
  static constexpr bool have_device = true;

  /// Constexpr variable that is true if the project is configured with CUDA support.
  static constexpr bool have_cuda = true;

  /**
   * @brief Check if a CUDA function call was successful and throw an exception if not.
   *
   * @param success Return value of a CUDA function call.
   * @param message An optional message to include in the exception.
   */
  inline void device_error_check(cudaError_t success, std::string message = "") {
    if (success != cudaSuccess) {
      std::cerr << "Cuda runtime error: " << std::to_string(success) << "\n"
                << " message: " << message << "\n"
                << " cudaGetErrorName: " << std::string(cudaGetErrorName(success)) << "\n"
                << " cudaGetErrorString: " << std::string(cudaGetErrorString(success)) << "\n";
#ifdef NDA_HAVE_MPI
      mpi::communicator{}.abort(31);
#else
      std::abort();
#endif
    }
  }

  /**
   * @brief Map between a single char and the corresponding `cublasOperation_t`.
   *
   * @details The mapping is as follows:
   * - 'N' -> `CUBLAS_OP_N` (non-transpose operation)
   * - 'T' -> `CUBLAS_OP_T` (transpose operation)
   * - 'C' -> `CUBLAS_OP_C` (conjugate transpose operation)
   * - everything else -> call `std::terminate()`.
   *
   * @param op Character to be mapped to a `cublasOperation_t`.
   * @return The corresponding `cublasOperation_t`.
   */
  inline cublasOperation_t get_cublas_op(char op) {
    switch (op) {
      case 'N': return CUBLAS_OP_N;
      case 'T': return CUBLAS_OP_T;
      case 'C': return CUBLAS_OP_C;
      default: std::terminate(); return {};
    }
  }

  template <typename T>
  inline auto to_cublas(T x) {
    return x;
  }

  /**
   * @brief Cast a `std::complex<double>` to a `cuDoubleComplex`.
   *
   * @param c `std::complex<double>` object.
   * @return Equivalent `cuDoubleComplex` object.
   */
  template <typename T>
  inline auto to_cublas(std::complex<T> c) {
    static_assert(std::is_same_v<float, std::remove_cvref_t<T>> or std::is_same_v<double, std::remove_cvref_t<T>>, "Error: Type mismatch.");
    if constexpr (std::is_same_v<double, std::remove_cvref_t<T>>)
      return cuDoubleComplex{c.real(), c.imag()};
    else if constexpr (std::is_same_v<float, std::remove_cvref_t<T>>)
      return cuComplex{c.real(), c.imag()};
  }

  /**
   * @brief Reinterpret a pointer to a `std::complex<double>` as a pointer to a `cuDoubleComplex`.
   *
   * @param c Pointer to a `std::complex<double>`.
   * @return Pointer to a `cuDoubleComplex` at the same address.
   */
  template <typename T>
  inline auto *to_cublas(std::complex<T> *c) {
    static_assert(std::is_same_v<float, std::remove_cvref_t<T>> or std::is_same_v<double, std::remove_cvref_t<T>>, "Error: Type mismatch.");
    if constexpr (std::is_same_v<double, std::remove_cvref_t<T>>)
      return reinterpret_cast<cuDoubleComplex *>(c); // NOLINT
    else if constexpr (std::is_same_v<float, std::remove_cvref_t<T>>)
      return reinterpret_cast<cuComplex *>(c); // NOLINT
  }

  /**
   * @brief Reinterpret a pointer to a `const std::complex<double>` as a pointer to a `const cuDoubleComplex`.
   *
   * @param c Pointer to a `const std::complex<double>`.
   * @return Pointer to a `const cuDoubleComplex` at the same address.
   */
  template <typename T>
  inline auto *to_cublas(std::complex<T> const *c) {
    static_assert(std::is_same_v<float, std::remove_cvref_t<T>> or std::is_same_v<double, std::remove_cvref_t<T>>, "Error: Type mismatch.");
    if constexpr (std::is_same_v<double, std::remove_cvref_t<T>>)
      return reinterpret_cast<const cuDoubleComplex *>(c); // NOLINT
    else if constexpr (std::is_same_v<float, std::remove_cvref_t<T>>)
      return reinterpret_cast<const cuComplex *>(c); // NOLINT
  }

  /**
   * @brief Reinterpret a pointer to a pointer to a `std::complex<double>` as a pointer to a pointer to a
   * `cuDoubleComplex`.
   *
   * @param c Pointer to a pointer to a `std::complex<double>`.
   * @return Pointer to a pointer to a `cuDoubleComplex` at the same address.
   */
  template <typename T>
  inline auto **to_cublas(std::complex<T> **c) {
    static_assert(std::is_same_v<float, std::remove_cvref_t<T>> or std::is_same_v<double, std::remove_cvref_t<T>>, "Error: Type mismatch.");
    if constexpr (std::is_same_v<double, std::remove_cvref_t<T>>)
      return reinterpret_cast<cuDoubleComplex **>(c); // NOLINT
    else if constexpr (std::is_same_v<float, std::remove_cvref_t<T>>)
      return reinterpret_cast<cuComplex **>(c); // NOLINT
  }

  /**
   * @brief Reinterpret a pointer to a pointer to a `const std::complex<double>` as a pointer to a pointer to a
   * `const cuDoubleComplex`.
   *
   * @param c Pointer to a pointer to a `const std::complex<double>`.
   * @return Pointer to a pointer to a `const cuDoubleComplex` at the same address.
   */
  template <typename T>
  inline auto **to_cublas(std::complex<T> const **c) {
    static_assert(std::is_same_v<float, std::remove_cvref_t<T>> or std::is_same_v<double, std::remove_cvref_t<T>>, "Error: Type mismatch.");
    if constexpr (std::is_same_v<double, std::remove_cvref_t<T>>)
      return reinterpret_cast<const cuDoubleComplex **>(c); // NOLINT
    else if constexpr (std::is_same_v<float, std::remove_cvref_t<T>>)
      return reinterpret_cast<const cuComplex **>(c); // NOLINT
  }

#else

/// Trigger a compilation error every time the nda::device_error_check function is called.
#define device_error_check(ARG1, ARG2) compile_error_no_gpu()

  /// Constexpr variable that is true if the project is configured with GPU support.
  static constexpr bool have_device = false;

  /// Constexpr variable that is true if the project is configured with CUDA support.
  static constexpr bool have_cuda = false;

#endif // NDA_HAVE_CUDA

  /** @} */

} // namespace nda
