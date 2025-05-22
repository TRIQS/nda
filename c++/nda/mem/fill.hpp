// Copyright (c) 2022-2023 Simons Foundation
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

#pragma once

#include "./address_space.hpp"
#include "../device.hpp"
#include "../traits.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <ranges>
#include <span>
#include <vector>

namespace nda::mem {

  /**
   * @brief Fill a range of memory with a specified value.
   *
   * @details The behaviour of the function depends on the address spaces:
   * - For `Host`, it simply calls `std::fill_n`.
   * - For `Device` and `Unified`, it calls `cudaMemset` or `cudaMemset2D` to transfer each byte of the value to the
   * destination memory.
   *
   * @tparam AdrSp nda::mem::AddressSpace of the destination.
   * @tparam T Value type.
   * @param first Pointer to the beginning of the destination memory.
   * @param count Number of elements to fill.
   * @param value Value to fill the memory with.
   * @return Pointer one past the last element filled.
   */
  template <AddressSpace AdrSp, typename T>
    requires(nda::is_scalar_or_convertible_v<T>)
  T *fill_n(T *first, size_t count, const T &value) {
    check_adr_sp_valid<AdrSp>();
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

    if constexpr (AdrSp == Host) {
      return std::fill_n(first, count, value);
    } else { // Device or Unified
      auto value_bytes = std::as_bytes(std::span(&value, 1));
      bool is_zero     = std::ranges::all_of(value_bytes, [](auto b) { return b == std::byte{0}; });
      if (is_zero) {
        device_error_check(cudaMemset(first, 0, count * sizeof(T)), "cudaMemset");
      } else {
        for (int n = 0; n < sizeof(T); ++n) {
          device_error_check(cudaMemset2D((char *)(first) + n, sizeof(T), static_cast<int>(value_bytes[n]), 1, count), "cudaMemset2D");
        }
      }
      return first + count;
    }
  }

  /**
   * @brief Fill a range of memory between two pointers with a specified value.
   *
   * @details It simply calls nda::mem::fill_n with the number of elements calculated from the pointers.
   *
   * @tparam AdrSp nda::mem::AddressSpace of the destination.
   * @tparam T Value type.
   * @param first Pointer to the beginning of the destination memory.
   * @param end Pointer to the end of the destination memory.
   * @param value Value to fill the memory with.
   * @return Pointer one past the last element filled.
   */
  template <AddressSpace AdrSp, typename T>
    requires(nda::is_scalar_or_convertible_v<T>)
  T *fill(T *first, T *end, const T &value) {
    if (std::distance(first, end) > 0) return fill_n<AdrSp>(first, std::distance(first, end), value);
    return first;
  }

  /**
   * @brief Fill a 2D memory region with a specified value.
   *
   * @details The behaviour of the function depends on the address spaces:
   * - For `Host`, the function is not implemented.
   * - For `Device` and `Unified`, it calls `cudaMemset2D` or `cudaMemcpy2D` to fill the 2D memory region with the 
   * specified value.
   *
   * @tparam AdrSp nda::mem::AddressSpace of the destination.
   * @tparam T Value type.
   * @param first Pointer to the beginning of the destination memory.
   * @param pitch Pitch of destination memory.
   * @param width Number of elements to fill in each row.
   * @param height Number of rows to fill.
   * @param value Value to fill the memory with.
   */
  template <AddressSpace AdrSp, typename T>
    requires(nda::is_scalar_or_convertible_v<T>)
  void fill2D_n(T *first [[maybe_unused]], size_t pitch [[maybe_unused]], size_t width, size_t height, const T &value) {
    check_adr_sp_valid<AdrSp>();
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");
    static_assert(AdrSp == mem::Device or AdrSp == mem::Unified, "Not implemented for host memory");

    bool is_zero = std::ranges::all_of(std::as_bytes(std::span(&value, 1)), [](auto b) { return b == std::byte{0}; });
    if (is_zero) {
      device_error_check(cudaMemset2D(first, pitch * sizeof(T), 0, width * sizeof(T), height), "cudaMemset2D");
    } else {
      std::vector<T> v(width * height, value);
      device_error_check(cudaMemcpy2D(first, pitch * sizeof(T), v.data(), width * sizeof(T), width * sizeof(T), height, cudaMemcpyDefault),
                         "cudaMemcpy2D");
    }
  }

} // namespace nda::mem
