// Copyright (c) 2023-2024 Simons Foundation
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
// Authors: Thomas Hahn, Geraud Krawezik, Nils Wentzell

/**
 * @file
 * @brief Provides GPU and non-GPU specific functionality.
 */

#pragma once

#ifdef NDA_HAVE_CUDA
#include "./exceptions.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <complex>
#include <exception>
#include <string>
#endif // NDA_HAVE_CUDA

#ifdef NDA_HAVE_ROCM
#include "hip/hip_runtime.h"
#include "hip/hip_complex.h"
#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver-extra-types.h"
#endif

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

  /// Constexpr variable that is true if the project is configured with ROCM support.
  static constexpr bool have_rocm = false;

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
   * @brief Cast a `std::complex<double>` to a `cuDoubleComplex`.
   *
   * @param c `std::complex<double>` object.
   * @return Equivalent `cuDoubleComplex` object.
   */
  inline auto cucplx(std::complex<double> c) { return cuDoubleComplex{c.real(), c.imag()}; }

  /**
   * @brief Reinterpret a pointer to a `std::complex<double>` as a pointer to a `cuDoubleComplex`.
   *
   * @param c Pointer to a `std::complex<double>`.
   * @return Pointer to a `cuDoubleComplex` at the same address.
   */
  inline auto *cucplx(std::complex<double> *c) { return reinterpret_cast<cuDoubleComplex *>(c); } // NOLINT

  /**
   * @brief Reinterpret a pointer to a `const std::complex<double>` as a pointer to a `const cuDoubleComplex`.
   *
   * @param c Pointer to a `const std::complex<double>`.
   * @return Pointer to a `const cuDoubleComplex` at the same address.
   */
  inline auto *cucplx(std::complex<double> const *c) { return reinterpret_cast<const cuDoubleComplex *>(c); } // NOLINT

  /**
   * @brief Reinterpret a pointer to a pointer to a `std::complex<double>` as a pointer to a pointer to a
   * `cuDoubleComplex`.
   *
   * @param c Pointer to a pointer to a `std::complex<double>`.
   * @return Pointer to a pointer to a `cuDoubleComplex` at the same address.
   */
  inline auto **cucplx(std::complex<double> **c) { return reinterpret_cast<cuDoubleComplex **>(c); } // NOLINT

  /**
   * @brief Reinterpret a pointer to a pointer to a `const std::complex<double>` as a pointer to a pointer to a
   * `const cuDoubleComplex`.
   *
   * @param c Pointer to a pointer to a `const std::complex<double>`.
   * @return Pointer to a pointer to a `const cuDoubleComplex` at the same address.
   */
  inline auto **cucplx(std::complex<double> const **c) { return reinterpret_cast<const cuDoubleComplex **>(c); } // NOLINT

#elif NDA_HAVE_ROCM

  /// Constexpr variable that is true if the project is configured with GPU support.
  static constexpr bool have_device = true;

  /// Constexpr variable that is true if the project is configured with CUDA support.
  static constexpr bool have_cuda = false;

  /// Constexpr variable that is true if the project is configured with ROCM support.
  static constexpr bool have_rocm = true;

  /**
   * @brief Check if a ROCM function call was successful and throw an exception if not.
   *
   * @param success Return value of a ROCM function call.
   * @param message An optional message to include in the exception.
   */
  inline void device_error_check(hipError_t success, std::string message = "") {
    if (success != hipSuccess) {
      NDA_RUNTIME_ERROR << "Rocm runtime error: " << std::to_string(success) << "\n"
                        << " message: " << message << "\n"
                        << " hipGetErrorName: " << std::string(hipGetErrorName(success)) << "\n"
                        << " hipGetErrorString: " << std::string(hipGetErrorString(success)) << "\n";
    }
  }

  /**
   * @brief Map between a single char and the corresponding `rocblas_operation`.
   *
   * @details The mapping is as follows:
   * - 'N' -> `rocblas_operation_none`
   * - 'T' -> `rocblas_operation_transpose`
   * - 'C' -> `rocblas_operation_conjugate_transpose`
   * - everything else -> call `std::terminate()`.
   *
   * @param op Character to be mapped to a `rocblas_operation`.
   * @return The corresponding `rocblas_operation`.
   */
  inline rocblas_operation get_rocblas_op(char op) {
    switch (op) {
      case 'N': return rocblas_operation_none;
      case 'T': return rocblas_operation_transpose;
      case 'C': return rocblas_operation_conjugate_transpose;
      default: std::terminate(); return {};
    }
  }

  /**
   * @brief Map between a single char and the corresponding `rocblas_svect`.
   *
   * @details The mapping is as follows:
   * - 'A' -> `rocblas_svect_all`
   * - 'S' -> `rocbals_svect_singular`
   * - 'O' -> `rocblas_svect_overwrite`
   * - 'N' -> `rocblas_svect_none`
   * - everything else -> call `std::terminate()`.
   *
   * @param op Character to be mapped to a `rocblas_svect`.
   * @return The corresponding `rocblas_svect`.
   */
  inline rocblas_svect get_rocblas_svect(char svect) {
    switch (svect) {
      case 'A': return rocblas_svect_all;
      case 'S': return rocblas_svect_singular;
      case 'O': return rocblas_svect_overwrite;
      case 'N': return rocblas_svect_none;
      default: std::terminate(); return {};
    }
  }

  /**
   * @brief Cast a `std::complex<double>` to a `rocblas_double_complex`.
   *
   * @param c `std::complex<double>` object.
   * @return Equivalent `rocblas_double_complex` object.
   */
  inline auto roccplx(std::complex<double> c) { return rocblas_double_complex{c.real(), c.imag()}; }

  /**
   * @brief Reinterpret a pointer to a `std::complex<double>` as a pointer to a `rocblas_double_complex`.
   *
   * @param c Pointer to a `std::complex<double>`.
   * @return Pointer to a `rocblas_double_complex` at the same address.
   */
  inline auto *roccplx(std::complex<double> *c) { return reinterpret_cast<rocblas_double_complex *>(c); } // NOLINT

  /**
   * @brief Reinterpret a pointer to a `const std::complex<double>` as a pointer to a `const rocblas_double_complex`.
   *
   * @param c Pointer to a `const std::complex<double>`.
   * @return Pointer to a `const rocblas_double_complex` at the same address.
   */
  inline auto *roccplx(std::complex<double> const *c) { return reinterpret_cast<const rocblas_double_complex *>(c); } // NOLINT

  /**
   * @brief Reinterpret a pointer to a pointer to a `std::complex<double>` as a pointer to a pointer to a
   * `rocblas_double_complex`.
   *
   * @param c Pointer to a pointer to a `std::complex<double>`.
   * @return Pointer to a pointer to a `rocblas_double_complex` at the same address.
   */
  inline auto **roccplx(std::complex<double> **c) { return reinterpret_cast<rocblas_double_complex **>(c); } // NOLINT

  /**
   * @brief Reinterpret a pointer to a pointer to a `const std::complex<double>` as a pointer to a pointer to a
   * `const rocblas_double_complex`.
   *
   * @param c Pointer to a pointer to a `const std::complex<double>`.
   * @return Pointer to a pointer to a `const rocblas_double_complex` at the same address.
   */
  inline auto **roccplx(std::complex<double> const **c) { return reinterpret_cast<const rocblas_double_complex **>(c); } // NOLINT

#else

/// Trigger a compilation error every time the nda::device_error_check function is called.
#define device_error_check(ARG1, ARG2) compile_error_no_gpu()

  /// Constexpr variable that is true if the project is configured with GPU support.
  static constexpr bool have_device = false;

  /// Constexpr variable that is true if the project is configured with CUDA support.
  static constexpr bool have_cuda = false;

  /// Constexpr variable that is true if the project is configured with ROCM support.
  static constexpr bool have_rocm = false;
#endif // NDA_HAVE_CUDA

  /** @} */

} // namespace nda
