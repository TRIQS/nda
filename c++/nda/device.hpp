// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides GPU and non-GPU specific functionality.
 */

#pragma once

#include <string_view>

#ifdef NDA_HAVE_CUDA
#include "./concepts.hpp"
#include "./exceptions.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <complex>
#include <exception>
#include <string>
#include <type_traits>
#endif // NDA_HAVE_CUDA

namespace nda {

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
      NDA_RUNTIME_ERROR << "Cuda runtime error: " << std::to_string(success) << "\n"
                        << " message: " << message << "\n"
                        << " cudaGetErrorName: " << std::string(cudaGetErrorName(success)) << "\n"
                        << " cudaGetErrorString: " << std::string(cudaGetErrorString(success)) << "\n";
    }
  }

  /**
   * @brief Synchronize the device.
   * 
   * @param do_sync If true, call `cudaDeviceSynchronize()` and check for errors. If false, do nothing.
   * @param func Optional name of the calling function to include in the error message.
   */
  inline void cuda_device_sync(bool do_sync = true, std::string_view func = "") {
    if (!do_sync) return;
    std::string msg = "cudaDeviceSynchronize failed";
    if (!func.empty()) {
      msg += " after call to ";
      msg.append(func);
    }
    device_error_check(cudaDeviceSynchronize(), std::move(msg));
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

  /**
   * @brief Type alias to map nda::FloatOrDouble types to their equivalent CUDA complex types.
   * @details Maps `float` to `cuComplex` and `double` to `cuDoubleComplex`.
   * @tparam T nda::FloatOrDouble type.
   */
  template <FloatOrDouble T>
  using cuda_complex_t = std::conditional_t<std::is_same_v<T, float>, cuComplex, cuDoubleComplex>;

  /**
   * @brief Cast a `std::complex<T>` to the equivalent CUDA type.
   * 
   * @details It casts 
   * - `std::complex<float>` to `cuComplex` and 
   * - `std::complex<double>` to `cuDoubleComplex`.
   *
   * @tparam T nda::FloatOrDouble type.
   * @param c `std::complex<T>` object.
   * @return Equivalent CUDA object.
   */
  template <FloatOrDouble T>
  cuda_complex_t<T> cucplx(std::complex<T> c) {
    return {c.real(), c.imag()};
  }

  /**
   * @brief Cast a pointer to a `std::complex<T>` to a pointer to the equivalent CUDA type.
   * 
   * @details It casts 
   * - `std::complex<float>*` to `cuComplex*` and 
   * - `std::complex<double>*` to `cuDoubleComplex*`.
   *
   * @tparam T nda::FloatOrDouble type.
   * @param c Pointer to a `std::complex<T>`.
   * @return Pointer to the equivalent CUDA type at the same address.
   */
  template <FloatOrDouble T>
  cuda_complex_t<T> *cucplx(std::complex<T> *c) {
    return reinterpret_cast<cuda_complex_t<T> *>(c); // NOLINT
  }

  /**
   * @brief Cast a pointer to a `const std::complex<T>` to a pointer to the equivalent CUDA type.
   * 
   * @details It casts 
   * - `const std::complex<float>*` to `const cuComplex*` and 
   * - `const std::complex<double>*` to `const cuDoubleComplex*`.
   *
   * @tparam T nda::FloatOrDouble type.
   * @param c Pointer to a `const std::complex<T>`.
   * @return Pointer to the equivalent CUDA type at the same address.
   */
  template <FloatOrDouble T>
  cuda_complex_t<T> const *cucplx(std::complex<T> const *c) {
    return reinterpret_cast<cuda_complex_t<T> const *>(c); // NOLINT
  }

  /**
   * @brief Cast a pointer to a pointer to a `std::complex<T>` to a pointer to a pointer to the equivalent CUDA type.
   * 
   * @details It casts
   * - `std::complex<float>**` to `cuComplex**` and
   * - `std::complex<double>**` to `cuDoubleComplex**`.
   *
   * @tparam T nda::FloatOrDouble type.
   * @param c Pointer to a pointer to a `std::complex<T>`.
   * @return Pointer to a pointer to the equivalent CUDA type at the same address.
   */
  template <FloatOrDouble T>
  cuda_complex_t<T> **cucplx(std::complex<T> **c) {
    return reinterpret_cast<cuda_complex_t<T> **>(c); // NOLINT
  }

  /**
   * @brief Cast a pointer to a pointer to a `const std::complex<double>` to a pointer to a pointer to the equivalent 
   * CUDA type.
   * 
   * @details It casts
   * - `const std::complex<float>**` to `const cuComplex**` and
   * - `const std::complex<double>**` to `const cuDoubleComplex**`.
   *
   * @tparam T nda::FloatOrDouble type.
   * @param c Pointer to a pointer to a `const std::complex<T>`.
   * @return Pointer to a pointer to the equivalent CUDA type at the same address.
   */
  template <FloatOrDouble T>
  cuda_complex_t<T> const **cucplx(std::complex<T> const **c) {
    return reinterpret_cast<cuda_complex_t<T> const **>(c); // NOLINT
  }

#else

/// Trigger a compilation error every time the nda::device_error_check function is called.
#define device_error_check(ARG1, ARG2) compile_error_no_gpu()

  /// Constexpr variable that is true if the project is configured with GPU support.
  static constexpr bool have_device = false;

  /// Constexpr variable that is true if the project is configured with CUDA support.
  static constexpr bool have_cuda = false;

  /// Empty function if `CudaSupport` is not enabled.
  inline void cuda_device_sync([[maybe_unused]] bool do_sync = true, [[maybe_unused]] std::string_view func = "") {}

#endif // NDA_HAVE_CUDA

  /** @} */

} // namespace nda
