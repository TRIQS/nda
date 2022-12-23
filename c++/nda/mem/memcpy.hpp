// Copyright (c) 2018-2021 Simons Foundation
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
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once

#include <cstring>

#include "address_space.hpp"
#include "../macros.hpp"
#include "../traits.hpp"
#include "device.hpp"

namespace nda::mem {

template <AddressSpace DestAdrSp, AddressSpace SrcAdrSp>
void memcpy(void *dest, void const *src, size_t count) {
  if constexpr (DestAdrSp == None or SrcAdrSp == None) {
    static_assert(always_false<bool>," memcpy<DestAdrSp == None or SrcAdrSp == None>: Oh Oh! ");
  } else if constexpr (DestAdrSp == Host && SrcAdrSp == Host) {
    std::memcpy(dest, src, count);
  } else {
#if defined(NDA_HAVE_CUDA)
    device_check( cudaMemcpy(dest, src, count, cudaMemcpyDefault), "CudaMemcpy" ); 
#else
    static_assert(always_false<bool>," Reached device code. Compile with GPU support."); 
#endif
  }
}

template <AddressSpace DestAdrSp, AddressSpace SrcAdrSp>
void memcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                        size_t width, size_t height)  {
  if constexpr (DestAdrSp == None or SrcAdrSp == None) {
    static_assert(always_false<bool>," memcpy2D<DestAdrSp == None or SrcAdrSp == None>: Oh Oh! ");
  } else if constexpr (DestAdrSp == Host && SrcAdrSp == Host) {
    for(size_t i=0; i<height; ++i, dst+=dpitch, src+=spitch)
      for(size_t j=0; j<width; ++j)
        *(dst+j) = *(src+j);
  } else {
#if defined(NDA_HAVE_CUDA)
    device_check( cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDefault), "CudaMemcpy2D" );
#else
    static_assert(always_false<bool>," Reached device code. Compile with GPU support.");
#endif
  }
}


} // namespace nda::mem
