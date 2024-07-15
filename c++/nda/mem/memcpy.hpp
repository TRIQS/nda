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
 * @brief Provides a generic memcpy and memcpy2D function for different address spaces.
 */

#pragma once

#include "./address_space.hpp"
#include "../device.hpp"
#include "../macros.hpp"

#include <cstring>

namespace nda::mem {

  /**
   * @addtogroup mem_utils
   * @{
   */

  /**
   * @brief Call the correct `memcpy` function based on the given address spaces.
   *
   * @details It makes the following function calls depending on the address spaces:
   * - `std::memcpy` if both address spaces are `Host`.
   * - `cudaMemcpy` for all other combinations.
   *
   * @tparam DestAdrSp nda::mem::AddressSpace of the destination.
   * @tparam SrcAdrSp nda::mem::AddressSpace of the source.
   * @param dest Pointer to the destination memory.
   * @param src Pointer to the source memory.
   * @param count Size in bytes to copy.
   */
  template <AddressSpace DestAdrSp, AddressSpace SrcAdrSp>
  void memcpy(void *dest, void const *src, size_t count) {
    check_adr_sp_valid<DestAdrSp, SrcAdrSp>();
    static_assert(nda::have_device == nda::have_cuda || nda::have_device == nda::have_rocm, "Adjust function for new device types");

    if constexpr (DestAdrSp == Host && SrcAdrSp == Host) {
      std::memcpy(dest, src, count);
    } else { // Device or Unified
#ifdef NDA_HAVE_CUDA
      device_error_check(cudaMemcpy(dest, src, count, cudaMemcpyDefault), "cudaMemcpy");
#elif NDA_HAVE_ROCM
      device_error_check(hipMemcpy(dest, src, count, hipMemcpyDefault), "hipMemcpy");
#endif
    }
  }

  /**
   * @brief Call CUDA's `cudaMemcpy2D` function or simulate its behavior on the `Host` based on the given address
   * spaces.
   *
   * @details Copies a matrix (`height` rows of `width` bytes each) from the memory area pointed to by `src` to the
   * memory area pointed to by `dest`. `dpitch` and `spitch` are the widths in memory in bytes of the 2D arrays pointed
   * to by `dest` and `src`, including any padding added to the end of each row. The memory areas may not overlap.
   * `width` must not exceed either `dpitch` or `spitch`.
   *
   * If both address spaces are `Host`, it simulates the behavior of CUDA's `cudaMemcpy2D` function by making multiple
   * calls to std::memcpy.
   *
   * @tparam DestAdrSp nda::mem::AddressSpace of the destination.
   * @tparam SrcAdrSp nda::mem::AddressSpace of the source.
   * @param dest Pointer to the destination memory.
   * @param dpitch Pitch of destination memory
   * @param src Pointer to the source memory.
   * @param spitch Pitch of source memory.
   * @param width Width of matrix transfer (columns in bytes).
   * @param height Height of matrix transfer (rows).
   */
  template <AddressSpace DestAdrSp, AddressSpace SrcAdrSp>
  void memcpy2D(void *dest, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height) {
    EXPECTS(width <= dpitch && width <= spitch);
    check_adr_sp_valid<DestAdrSp, SrcAdrSp>();
    static_assert(nda::have_device == nda::have_cuda || nda::have_device == nda::have_rocm, "Adjust function for new device types");

    if constexpr (DestAdrSp == Host && SrcAdrSp == Host) {
      auto *desti = static_cast<unsigned char *>(dest);
      auto *srci  = static_cast<const unsigned char *>(src);
      for (size_t i = 0; i < height; ++i, desti += dpitch, srci += spitch) std::memcpy(desti, srci, width);
    } else { // Device or Unified
#ifdef NDA_HAVE_CUDA
      device_error_check(cudaMemcpy2D(dest, dpitch, src, spitch, width, height, cudaMemcpyDefault), "cudaMemcpy2D");
#elif NDA_HAVE_ROCM
      device_error_check(hipMemcpy2D(dest, dpitch, src, spitch, width, height, hipMemcpyDefault), "hipMemcpy2D");
#endif
    }
  }

  /** @} */

} // namespace nda::mem
