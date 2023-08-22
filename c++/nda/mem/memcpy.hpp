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

#include <cstring>

#include "address_space.hpp"
#include "../macros.hpp"
#include "../traits.hpp"

namespace nda::mem {

  template <AddressSpace DestAdrSp, AddressSpace SrcAdrSp>
  void memcpy(void *dest, void const *src, size_t count) {
    check_adr_sp_valid<DestAdrSp, SrcAdrSp>();
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

    if constexpr (DestAdrSp == Host && SrcAdrSp == Host) {
      std::memcpy(dest, src, count);
    } else { // Device or Unified
      device_error_check(cudaMemcpy(dest, src, count, cudaMemcpyDefault), "CudaMemcpy");
    }
  }

  // FIXMEOP Comment the meaning of inputs
  // What is pitch ?
  template <AddressSpace DestAdrSp, AddressSpace SrcAdrSp>
  void memcpy2D(void *dest, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height) {
    EXPECTS(width <= dpitch && width <= spitch);
    check_adr_sp_valid<DestAdrSp, SrcAdrSp>();
    static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

    if constexpr (DestAdrSp == Host && SrcAdrSp == Host) {
      auto *desti = static_cast<unsigned char *>(dest);
      auto *srci  = static_cast<const unsigned char *>(src);
      for (size_t i = 0; i < height; ++i, desti += dpitch, srci += spitch) std::memcpy(desti, srci, width);
    } else if (nda::have_device) {
      device_error_check(cudaMemcpy2D(dest, dpitch, src, spitch, width, height, cudaMemcpyDefault), "CudaMemcpy2D");
    }
  }

} // namespace nda::mem
