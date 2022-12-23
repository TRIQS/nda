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

#include <cstdlib> 
#include <algorithm>

#include "address_space.hpp"
#include "../macros.hpp"
#include "../traits.hpp"
#include "device.hpp"

namespace nda::mem {

template <AddressSpace AdrSp>
void memset(void* p, int value, size_t count)
{
  if constexpr (AdrSp == Host) {
    std::memset(p,value,count);
  } else if constexpr (AdrSp == Device or AdrSp == Unified) {
#if defined(NDA_HAVE_CUDA)
    device_check( cudaMemset(p, value, count), "cudaMemset" );
#else
    static_assert(always_false<bool>," Reached device code. Compile with GPU support.");
#endif
  } else if constexpr (AdrSp == None) {
    static_assert(always_false<bool>," malloc<AdrSp == None>: Oh Oh! ");
  }
}

template <AddressSpace AdrSp>
void memset2D(void* ptr, size_t pitch, int value, size_t width, size_t height) 
{ 
  if constexpr (AdrSp == Host) {
    auto v = static_cast<unsigned char>(value);
    unsigned char* p = reinterpret_cast<unsigned char*>(ptr);
    for(size_t i=0; i<height; ++i, p+=pitch) 
      for(size_t j=0; j<width; ++j) 
        *(p+j) = v;
  } else if constexpr (AdrSp == Device or AdrSp == Unified) {
#if defined(NDA_HAVE_CUDA)
    device_check( cudaMemset2D(ptr, pitch, value, width, height), "cudaMemset2D" );
#else
    static_assert(always_false<bool>," Reached device code. Compile with GPU support.");
#endif
  } else if constexpr (AdrSp == None) {
    static_assert(always_false<bool>," malloc<AdrSp == None>: Oh Oh! ");
  }
}

} // namespace nda::mem

