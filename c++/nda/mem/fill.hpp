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
#include "../macros.hpp"
#include "../traits.hpp"

namespace nda::mem {

  template <AddressSpace AdrSp, typename T>
  T *fill_n(T *first, size_t count, const T &value) {
    check_adr_sp_valid<AdrSp>();
    static_assert(nda::is_scalar_or_convertible_v<T>, "Incompatible scalar type");
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

    if constexpr (AdrSp == Host) {
      return std::fill_n(first, count, value);
    } else { // Device or Unified
      if (std::find_if((char const *)(&value), (char const *)(&value) + sizeof(T), [](char c) { return c != 0; })
          == (char const *)(&value) + sizeof(T)) {
        device_error_check(cudaMemset((void *)first, 0, count * sizeof(T)), "cudaMemset");
      } else {
        // MAM: temporary, use kernel/thrust/foreach/... when available
        int v             = 0;
        uint8_t const *ui = reinterpret_cast<uint8_t const *>(&value);
        uint8_t *fn       = reinterpret_cast<uint8_t *>(first);
        for (int n = 0; n < sizeof(T); ++n) {
          v = 0; // just in case
          v = *(ui + n);
          device_error_check(cudaMemset2D((void *)(fn + n), sizeof(T), v, 1, count), "cudaMemset2D");
        }
      }
      return first + count;
    }
  }

  template <AddressSpace AdrSp, typename T>
    requires(nda::is_scalar_or_convertible_v<T>)
  T *fill(T *first, T *end, const T &value) {
    check_adr_sp_valid<AdrSp>();
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

    if (std::distance(first, end) > 0) return fill_n<AdrSp>(first, std::distance(first, end), value);
    return first;
  }

  template <AddressSpace AdrSp, typename T>
    requires(nda::is_scalar_or_convertible_v<T>)
  void fill2D_n(T *first, size_t pitch, size_t width, size_t height, const T &value) {
    check_adr_sp_valid<AdrSp>();
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

    if constexpr (AdrSp == Host) {
    } else { // Device or Unified
      if (std::find_if((char const *)(&value), (char const *)(&value) + sizeof(T), [](char c) { return c != 0; })
          == (char const *)(&value) + sizeof(T)) {
        device_error_check(cudaMemset2D((void *)first, pitch * sizeof(T), 0, width * sizeof(T), height), "cudaMemset2D");
      } else {
        // MAM: temporary, use kernel/thrust/foreach/... when available
        // as a temporary version, can also loop over rows...
        std::vector<T> v(width * height, value);
        device_error_check(
           cudaMemcpy2D((void *)first, pitch * sizeof(T), (void *)v.data(), width * sizeof(T), width * sizeof(T), height, cudaMemcpyDefault),
           "cudaMemcpy2D");
      }
    }
  }

} // namespace nda::mem
