// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic malloc and free function for different address spaces.
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
   * @brief Call the correct `malloc` function based on the given address space.
   *
   * @details It makes the following function calls depending on the address space:
   * - `std::malloc` for `Host`.
   * - `cudaMalloc` for `Device`.
   * - `cudaMallocManaged` for `Unified`.
   *
   * @tparam AdrSp nda::mem::AddressSpace.
   * @param size Size in bytes to be allocated.
   * @return Pointer to the allocated memory.
   */
  template <AddressSpace AdrSp>
  void *malloc(size_t size) {
    check_adr_sp_valid<AdrSp>();
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

    void *ptr = nullptr;
    if constexpr (AdrSp == Host) {
      ptr = std::malloc(size); // NOLINT (we want to return a void*)
    } else if constexpr (AdrSp == Device) { // NOLINT (branch is not repeated)
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
  void free(void *p) {
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
