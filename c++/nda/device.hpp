// Copyright (c) 2023 Simons Foundation
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
// Authors: Nils Wentzell

#pragma once

#include <iostream>
#include "exceptions.hpp"

#ifdef NDA_HAVE_CUDA
#include "cuda_runtime.h"
#include "cublas_v2.h"
#endif

namespace nda {

  // Use within constexpr branch to trigger compiler error
  template <bool flag = false>
  void compile_error_no_gpu() {
    static_assert(flag, "Using device functionality without gpu support! Configure project with -DSupportGPU=ON.");
  }

#ifdef NDA_HAVE_CUDA
  static constexpr bool have_device = true;
  static constexpr bool have_cuda   = true;

  inline void device_error_check(cudaError_t success, std::string message = "") {
    if (success != cudaSuccess) {
      NDA_RUNTIME_ERROR << "Cuda runtime error: " << std::to_string(success) << "\n"
                        << " message: " << message << "\n"
                        << " cudaGetErrorName: " << std::string(cudaGetErrorName(success)) << "\n"
                        << " cudaGetErrorString: " << std::string(cudaGetErrorString(success)) << "\n";
    }
  }

  inline cublasOperation_t get_cublas_op(char op) {
    switch (op) {
      case 'N': return CUBLAS_OP_N;
      case 'T': return CUBLAS_OP_T;
      case 'C': return CUBLAS_OP_C;
      default: std::terminate(); return {};
    }
  }

  inline auto cucplx(std::complex<double> c) { return cuDoubleComplex{c.real(), c.imag()}; }
  inline auto *cucplx(std::complex<double> *c) { return reinterpret_cast<cuDoubleComplex *>(c); }                // NOLINT
  inline auto *cucplx(std::complex<double> const *c) { return reinterpret_cast<const cuDoubleComplex *>(c); }    // NOLINT
  inline auto **cucplx(std::complex<double> **c) { return reinterpret_cast<cuDoubleComplex **>(c); }             // NOLINT
  inline auto **cucplx(std::complex<double> const **c) { return reinterpret_cast<const cuDoubleComplex **>(c); } // NOLINT
#else

#define device_error_check(ARG1, ARG2) compile_error_no_gpu()
  static constexpr bool have_device = false;
  static constexpr bool have_cuda   = false;
#endif

} // namespace nda
