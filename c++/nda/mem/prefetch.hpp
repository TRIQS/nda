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

namespace nda::mem {

  // MAM: should I keep a specific stream just for prefetching???
  template <AddressSpace AdrSp>
  void prefetch(void *p, size_t count) {
    static_assert(AdrSp != None and AdrSp != Unified);
#if defined(NDA_HAVE_CUDA)
    if constexpr (AdrSp == Host)
      device_check(cudaMemPrefetchAsync(p, count, cudaCpuDeviceId, 0), "cudaMemPrefetchAsync");
    else if constexpr (AdrSp == Device) {
      int dev = 0;
      device_check(cudaGetDevice(&dev), "cudagetDevice");
      device_check(cudaMemPrefetchAsync(p, count, dev, 0), "cudaMemPrefetchAsync");
    }
#else
    compile_error_no_gpu();
#endif
  }

} // namespace nda::mem
