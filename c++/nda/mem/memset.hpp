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

#include "address_space.hpp"
#include "../macros.hpp"
#include "../traits.hpp"

namespace nda::mem {

  template <AddressSpace AdrSp>
  void memset(void *p, int value, size_t count) {
    check_adr_sp_valid<AdrSp>();
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

    if constexpr (AdrSp == Host) {
      std::memset(p, value, count);
    } else { // Device or Unified
      device_error_check(cudaMemset(p, value, count), "cudaMemset");
    }
  }

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

} // namespace nda::mem
