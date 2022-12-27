// Copyright (c) 2019-2021 Simons Foundation
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
#include <complex>
#include <string_view>
#include "nda/exceptions.hpp"
#include "nda/traits.hpp"
#include "nda/declarations.hpp"
#include "nda/mem/address_space.hpp"
#include "nda/mem/device.hpp"
#include "nda/mem/malloc.hpp"
#include "nda/mem/memcpy.hpp"

#if defined(NDA_HAVE_TBLIS)
#include "interface/tblis_interface.hpp"
#endif

#if defined(NDA_HAVE_CUTENSOR)
#include "interface/cutensor_interface.hpp"
#endif

namespace nda::tensor {

  /**
   * Compute a(...) = alpha 
   */
  template <MemoryArray A>
  requires(is_blas_lapack_v<get_value_t<A>>) 
  void set(get_value_t<A> alpha, A &&a) {

    using value_t = get_value_t<A>;
    constexpr int rank = get_rank<A>;

    if constexpr (mem::on_host<A>) {
      a() = alpha;  // is there a point in using tblis? 
    } else { // on device
#if defined(NDA_HAVE_CUTENSOR)
      std::string indx = default_index<uint8_t(rank)>();
      cutensor::cutensor_desc<value_t,rank> a_t(a,op::ID);
      value_t* z;
      mem::device_check( cudaMalloc((void**) &z, sizeof(value_t)), "CudaMalloc" );
      mem::device_check( cudaMemcpy((void*) z, (void*) &alpha, sizeof(value_t), cudaMemcpyDefault), "CudaMemcpy" );
      cutensor::cutensor_desc<value_t,0> z_t(z,op::ID);
      cutensor::elementwise_binary(value_t{1},z_t,z,"",
                                   value_t{0},a_t,a.data(),indx,
                                   a.data(),op::SUM);
      mem::device_check( cudaFree((void*)z), "cudaFree" );
#else
      static_assert(always_false<bool>," set on device requires gpu tensor operations backend. ");
#endif
    }
  }

} // namespace nda::tensor
