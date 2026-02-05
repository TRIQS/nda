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
// Authors: Miguel Morales, Nils Wentzell

/**
 * @file
 * @brief Provides an interface to CUDA's `cudaMemPrefetchAsync` routine.
 */

#pragma once

#include "./address_space.hpp"
#include "../device.hpp"

#include <cstdlib>

namespace nda::mem {

  /**
   * @addtogroup mem_utils
   * @{
   */

  // MAM: should I keep a specific stream just for prefetching???
  /**
   * @brief Prefetch memory to the specified destination location. 
   *
   * @details It makes a call to `cudaMemPrefetchAsync` to prefetch memory to either the `Host` or `Device` address 
   * space.
   *
   * @tparam AdrSp nda::mem::AddressSpace (either `Host` or `Device`).
   * @param p Pointer to the memory to prefetch.
   * @param count Number of bytes to prefetch.
   */
  template <AddressSpace AdrSp>
    requires((AdrSp == Host or AdrSp == Device) and have_cuda)
  void prefetch(void *p, size_t count) {
    if constexpr (AdrSp == Host) {
      device_error_check(cudaMemPrefetchAsync(p, count, cudaCpuDeviceId, 0), "cudaMemPrefetchAsync");
    } else if constexpr (AdrSp == Device) {
      int dev = 0;
      device_error_check(cudaGetDevice(&dev), "cudagetDevice");
      device_error_check(cudaMemPrefetchAsync(p, count, dev, 0), "cudaMemPrefetchAsync");
    }
  }

  /** @} */

} // namespace nda::mem
