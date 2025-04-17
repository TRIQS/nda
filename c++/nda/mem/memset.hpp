// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic `memset` and `memset2D` function for different address spaces.
 */

#pragma once

#include "address_space.hpp"
#include "../device.hpp"

#include <cstring>

namespace nda::mem {

  /**
   * @addtogroup mem_utils
   * @{
   */

  /**
   * @brief Call the correct `memset` function based on the given address space.
   *
   * @details It makes the following function calls depending on the address spaces:
   * - `std::memset` for `Host`.
   * - `cudaMemset` for `Device` and `Unified`.
   *
   * @tparam AdrSp nda::mem::AddressSpace.
   * @param p Pointer to the memory to be set.
   * @param value Value to set for each byte of specified memory.
   * @param count Size in bytes to be set.
   */
  template <AddressSpace AdrSp>
  void memset(void *p, int value, size_t count) {
    check_adr_sp_valid<AdrSp>();
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

    if constexpr (AdrSp == Host) {
      std::memset(p, value, count);
    } else {
      device_error_check(cudaMemset(p, value, count), "cudaMemset");
    }
  }

  /**
   * @brief Call CUDA's `cudaMemset2D` function or simulate its behavior on the `Host` based on the given address space.
   *
   * @details Sets each byte of a matrix (`height` rows of `width` bytes each) pointed to by `dest` to a specified
   * value. `pitch` is the width in memory in bytes of the 2D array pointed to by `dest`, including any padding added to
   * the end of each row.
   *
   * If the address space is `Host`, it simulates the behavior of CUDA's `cudaMemset2D` function by making multiple
   * calls to std::memset.
   *
   * @tparam AdrSp nda::mem::AddressSpace.
   * @param ptr Pointer to the memory to be set.
   * @param pitch Pitch in bytes of the memory (unused if height is 1).
   * @param value Value to set for each byte of specified memory
   * @param width Width of matrix to set (columns in bytes).
   * @param height Height of matrix to set (rows).
   */
  template <AddressSpace AdrSp>
  void memset2D(void *ptr, size_t pitch, int value, size_t width, size_t height) {
    check_adr_sp_valid<AdrSp>();
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

    if constexpr (AdrSp == Host) {
      auto *ptri = static_cast<unsigned char *>(ptr);
      for (size_t i = 0; i < height; ++i, ptri += pitch) std::memset(ptri, value, width);
    } else { // Device or Unified
      device_error_check(cudaMemset2D(ptr, pitch, value, width, height), "cudaMemset2D");
    }
  }

  /** @} */

} // namespace nda::mem
