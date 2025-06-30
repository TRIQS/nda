// Copyright (c) 2022-2024 Simons Foundation
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
// Authors: Thomas Hahn, Miguel Morales, Nils Wentzell

/**
 * @file
 * @brief Provides a generic aligned_alloc and aligned_free function for different address spaces.
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

  /**
   * @brief Call the correct `aligned_alloc` function based on the given address space.
   *
   * @details It makes the following function calls depending on the address space:
   * - `std::aligned_alloc` for `Host`.
   * - `cudaMalloc` for `Device`.
   * - `cudaMallocManaged` for `Unified`.
   *
   * @tparam AdrSp nda::mem::AddressSpace.
   * @param size Size in bytes to be allocated.
   * @return Pointer to the allocated memory.
   */
  template <AddressSpace AdrSp>
  void *aligned_alloc(size_t alignment, size_t size) {
    check_adr_sp_valid<AdrSp>();
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

    void *ptr = nullptr;
    if constexpr (AdrSp == Host) {
      if (alignment != 0) {
        size = next_multiple(size, alignment);
      }
      if (alignment >= 8UL) {
        ptr = std::aligned_alloc(alignment, size);
      } else {
        ptr = std::malloc(size); // NOLINT (we want to return a void*)
      }
    } else if constexpr (AdrSp == Device) {
      device_error_check(cudaMalloc((void **)&ptr, size), "cudaMalloc");
    } else {
      device_error_check(cudaMallocManaged((void **)&ptr, size), "cudaMallocManaged");
    }
    return ptr;
  }

  /**
   * @brief Call the correct `free` function based on the given address space.
   *
   * @details It makes the following function calls depending on the address space:
   * - `std::free` for `Host`.
   * - `cudaFree` for `Device` and `Unified`.
   *
   * @tparam AdrSp nda::mem::AddressSpace.
   * @param p Pointer to the memory to be freed.
   */
  template <AddressSpace AdrSp>
  void aligned_free(void *p) {
    check_adr_sp_valid<AdrSp>();
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

    if constexpr (AdrSp == Host) {
      std::free(p); // NOLINT (we want to call free with a void*)
    } else {
      device_error_check(cudaFree(p), "cudaFree");
    }
  }

  /** @} */

} // namespace nda::mem
