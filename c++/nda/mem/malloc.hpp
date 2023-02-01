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

namespace nda::mem {

template <AddressSpace AdrSp>
void* malloc(size_t size) {
  check_adr_sp_valid<AdrSp>();
  static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

  if constexpr (AdrSp == Host) {
    return std::malloc(size);
  } else if constexpr (AdrSp == Device) {
    void* ptr = nullptr;
    device_check( cudaMalloc((void**)&ptr, size), "cudaMalloc" ); 
    return ptr;
  } else { // Unified
    void* ptr = nullptr;
    device_check( cudaMallocManaged((void**)&ptr, size), "cudaMallocManaged" );
    return ptr;
  }
  return nullptr;
}

template <AddressSpace AdrSp>
void free(void* p) {
  check_adr_sp_valid<AdrSp>();
  static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

  if constexpr (AdrSp == Host) {
    std::free(p);
  } else { // Device or Unified
    device_check( cudaFree(p), "cudaFree" );
  } 
}

template <AddressSpace AdrSp>
void* calloc(size_t num, size_t size) {
  check_adr_sp_valid<AdrSp>();
  static_assert(nda::have_device == nda::have_cuda, "Adjust function for new device types");

  if constexpr (AdrSp == Host) {
    return std::calloc(num,size);
  } else { // Device or Unified
    char* ptr = (char*) malloc<AdrSp>(num*size);
    device_check( cudaMemset((void*)ptr,0,num*size), "cudaMemset" );
    return (void*)ptr;
  }
}

} // namespace nda::mem
