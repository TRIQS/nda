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

#include "address_space.hpp"
#include "../macros.hpp"
#include "../traits.hpp"
#include "device.hpp"
//#include "fill.hpp"

namespace nda::mem {

template <AddressSpace AdrSp>
void* malloc(size_t size) {
  if constexpr (AdrSp == Host) {
    return std::malloc(size);
  } else if constexpr (AdrSp == Device) {
#if defined(NDA_HAVE_CUDA)
    void* ptr = nullptr;
    device_check( cudaMalloc((void**)&ptr, size), "cudaMalloc" ); 
    return ptr;
#else
    static_assert(always_false<bool>," Reached device code. Compile with GPU support."); 
#endif
  } else if constexpr (AdrSp == Unified) {
#if defined(NDA_HAVE_CUDA)
    void* ptr = nullptr;
    device_check( cudaMallocManaged((void**)&ptr, size), "cudaMallocManaged" );                 
    return ptr;
#else
    static_assert(always_false<bool>," Reached device code. Compile with GPU support.");
#endif
  } else if constexpr (AdrSp == None) {
    static_assert(always_false<bool>," malloc<AdrSp == None>: Oh Oh! "); 
  }
  return nullptr;
}

template <AddressSpace AdrSp>
void free(void* p) {
  if constexpr (AdrSp == Host) {
    std::free(p);
  } else if constexpr (AdrSp == Device or AdrSp == Unified) {
#if defined(NDA_HAVE_CUDA)
    device_check( cudaFree(p), "cudaFree" );
#else
    static_assert(always_false<bool>," Reached device code. Compile with GPU support.");
#endif
  } else if constexpr (AdrSp == None) {
    static_assert(always_false<bool>," malloc<AdrSp == None>: Oh Oh! ");
  }
}

template <AddressSpace AdrSp>
void* calloc(size_t num, size_t size) {
  if constexpr (AdrSp == Host) {
    return std::calloc(num,size);
  } else if constexpr (AdrSp == Device or AdrSp == Unified) {
    char* ptr = (char*) malloc<AdrSp>(num*size);
#if defined(NDA_HAVE_CUDA)
    device_check( cudaMemset((void*)ptr,0,num*size), "cudaMemset" );
#else
    static_assert(always_false<bool>," Reached device code. Compile with GPU support.");
#endif
//    fill<AdrSp>(ptr, 0, num*size);
    return (void*)ptr;
  } else if constexpr (AdrSp == None) {
    static_assert(always_false<bool>," malloc<AdrSp == None>: Oh Oh! ");
  }
  return nullptr;
}

} // namespace nda::mem
