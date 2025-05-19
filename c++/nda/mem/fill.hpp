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

#include <cstdlib>
#include <algorithm>
#include <vector>

#include "address_space.hpp"
#include "../traits.hpp"

namespace nda::mem {

  /**
   * @brief Fills a range of memory with a specified value.
   *
   * The behavior depends on the AddressSpace (Host, Device, or Unified).
   *
   * @tparam AdrSp The address space (e.g., Host, Device, Unified).
   * @tparam T The type of the elements to fill.
   * @param first Pointer to the beginning of the range.
   * @param count Number of elements to fill.
   * @param value The value to fill the range with.
   * @return Pointer to the end of the filled range.
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
      bool is_zero     = std::ranges::equal(value_bytes, std::views::repeat(std::byte{0}));
      if (is_zero) {
        device_error_check(cudaMemset(first, 0, count * sizeof(T)), "cudaMemset");
      } else {
        for (int n = 0; n < sizeof(T); ++n) {
          const int byte_value = static_cast<int>(value_bytes[n]);
          device_error_check(cudaMemset2D((char *)(first) + n, sizeof(T), byte_value, 1, count), "cudaMemset2D");
        }
      }
      return first + count;
    }
  }

  /**
   * @brief Fills a range of memory between two pointers with a specified value.
   *
   * Internally calls `fill_n`.
   *
   * @tparam AdrSp The address space (e.g., Host, Device, Unified).
   * @tparam T The type of the elements to fill.
   * @param first Pointer to the beginning of the range.
   * @param end Pointer to the end of the range.
   * @param value The value to fill the range with.
   * @return Pointer to the end of the filled range.
   */
  template <AddressSpace AdrSp, typename T>
    requires(nda::is_scalar_or_convertible_v<T>)
  T *fill(T *first, T *end, const T &value) {
    if (std::distance(first, end) > 0) return fill_n<AdrSp>(first, std::distance(first, end), value);
    return first;
  }

  /**
   * @brief Fills a 2D memory region with a specified value.
   *
   * The behavior depends on the AddressSpace (Host, Device, or Unified).
   *
   * @tparam AdrSp The address space (e.g., Host, Device, Unified).
   * @tparam T The type of the elements to fill.
   * @param first Pointer to the beginning of the 2D memory region.
   * @param pitch The memory pitch between rows.
   * @param width The number of elements to fill in each row.
   * @param height The number of rows to fill.
   * @param value The value to fill the 2D region with.
   */
  template <AddressSpace AdrSp, typename T>
    requires(nda::is_scalar_or_convertible_v<T>)
  void fill2D_n(T *first, size_t pitch, size_t width, size_t height, const T &value) {
    check_adr_sp_valid<AdrSp>();
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");
    static_assert(AdrSp == mem::Device or AdrSp == mem::Unified, "Not implemented for host memory");

    bool is_zero = std::ranges::equal(std::as_bytes(std::span(&value, 1)), std::views::repeat(std::byte{0}));
    if (is_zero) {
      device_error_check(cudaMemset2D(first, pitch * sizeof(T), 0, width * sizeof(T), height), "cudaMemset2D");
    } else {
      std::vector<T> v(width * height, value);
      device_error_check(cudaMemcpy2D(first, pitch * sizeof(T), v.data(), width * sizeof(T), width * sizeof(T), height, cudaMemcpyDefault),
                         "cudaMemcpy2D");
    }
  }

} // namespace nda::mem
