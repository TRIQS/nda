// Copyright (c) 2018-2021 Simons Foundation
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
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once

#include <iostream>
#include "nda/exceptions.hpp"

#if defined(NDA_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

namespace nda {

  // Use within constexpr branch to trigger compiler error
  template<bool flag = false> void compile_error_no_gpu() { static_assert(flag, "Using device functionality without gpu support! Configure project with -DSupportGPU=ON."); }

#if defined(NDA_HAVE_CUDA)
  static constexpr bool have_device = true;
  static constexpr bool have_cuda   = true;

  inline void device_check(cudaError_t sucess, std::string message = "") {
    if (sucess != cudaSuccess) {
      NDA_RUNTIME_ERROR << "Cuda runtime error: " << std::to_string(sucess) << "\n"
                        << " message: " << message << "\n"
                        << " cudaGetErrorName: " << std::string(cudaGetErrorName(sucess)) << "\n"
                        << " cudaGetErrorString: " << std::string(cudaGetErrorString(sucess)) << "\n";
    }
  }

#else

#define device_check(ARG1, ARG2) compile_error_no_gpu()
  static constexpr bool have_device = false;
  static constexpr bool have_cuda   = false;
#endif

} // namespace nda
