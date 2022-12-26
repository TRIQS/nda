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
#include "../exceptions.hpp"
#include "../traits.hpp"
#include "../declarations.hpp"
#include "../mem/address_space.hpp"

#if defined(NDA_HAVE_TBLIS)
#include "interface/tblis_interface.hpp"
#endif

#if defined(NDA_HAVE_CUTENSOR)
#include "interface/cutensor_interface.hpp"
#endif

namespace nda::tensor {

  /**
   * Computes x = op(A(...)), where op = {std::plus<>{},std::max<>{},std::min<>{},...}. 
   */
  template <MemoryArray A>
  requires(is_blas_lapack_v<get_value_t<A>>) 
  get_value_t<A> reduce(A &&a, op::TENSOR_OP oper = op::SUM) {

    using value_t = get_value_t<A>;
    constexpr int rank = get_rank<A>;

    if constexpr (mem::on_host<A>) {
#if defined(NDA_HAVE_TBLIS)
      nda_tblis::tensor<value_t,rank> a_t(a);
      std::string indx = default_index<uint8_t(rank)>(); 
      nda_tblis::scalar<value_t> res(0);
      std::array<::tblis::len_type, rank> idx; 
      // MAM: do this in a cleaner way!
      if( oper == op::SUM ) { 
        ::tblis::tblis_tensor_reduce(NULL,NULL,::tblis::REDUCE_SUM,&a_t,indx.data(),&res,idx.data());
      } else if( oper == op::MAX ) {
        ::tblis::tblis_tensor_reduce(NULL,NULL,::tblis::REDUCE_MAX,&a_t,indx.data(),&res,idx.data());
      } else if( oper == op::MIN ) {
        ::tblis::tblis_tensor_reduce(NULL,NULL,::tblis::REDUCE_MIN,&a_t,indx.data(),&res,idx.data());
      } else {
        NDA_RUNTIME_ERROR <<"tensor::reduce: Unknown reduction operation."; 
      }
      return res.value();
#else
      static_assert(always_false<bool>," reduce on host requires cpu tensor operations backend. ");
#endif
    } else { // on device
#if defined(NDA_HAVE_CUTENSOR)
//      cutensor::termbyterm();
      static_assert(always_false<bool>," reduce on device cuTensor!!!. ");
#else
      static_assert(always_false<bool>," reduce on device requires gpu tensor operations backend. ");
#endif
    }
  }

} // namespace nda::tensor
