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

#if defined(NDA_HAVE_TBLIS)
#include "interface/tblis_interface.hpp"
#endif

#if defined(NDA_HAVE_CUTENSOR)
#include "interface/cutensor_interface.hpp"
#endif

namespace nda::tensor {

  /**
   * Compute A(...) = alpha * A(...) 
   */
  template <MemoryArray A>
  requires(is_blas_lapack_v<get_value_t<A>>) 
  void scale(get_value_t<A> alpha, A &&a) {

    using value_t = get_value_t<A>;
    constexpr int rank = get_rank<A>;

    if constexpr (mem::on_host<A>) {
//#if defined(NDA_HAVE_TBLIS)
//      nda_tblis::tensor<value_t,get_rank<A>> a_t(a,alpha);
//      std::string indx = default_index<uint8_t(get_rank<A>)>(); 
//      ::tblis::tblis_tensor_scale(NULL,NULL,&a_t,indx.data());
//#else
      a() *= alpha;
//#endif
    } else { // on device
#if defined(NDA_HAVE_CUTENSOR)
      cutensor::cutensor_desc<value_t,rank> a_t(a,op::ID);
      std::string indx = default_index<uint8_t(rank)>();
      cutensor::permute(alpha, a_t, a.data(), indx, a_t, a.data(), indx);
#else
      static_assert(always_false<bool>," scale on device requires gpu tensor operations backend. ");
#endif
    }
  }

  template <MemoryArray A>
  requires(is_blas_lapack_v<get_value_t<A>>)
  void scale(get_value_t<A> alpha, A &&a, op::TENSOR_OP oper) {

    using value_t = get_value_t<A>;
    constexpr int rank = get_rank<A>;

    if constexpr (mem::on_host<A>) {
      switch(oper) {
        case op::ID:
	  a() *= alpha;
	  break;
        case op::CONJ:
          a = nda::conj(a)*alpha;
	  break;
        case op::SQRT:
          a = nda::sqrt(a)*alpha;
	  break;
        case op::ABS:
          a = nda::abs(a)*alpha;
	  break;
        case op::NEG:
          a = a*(-1.0*alpha);
	  break;
        default:
     	 NDA_RUNTIME_ERROR <<"Unknown unary operation."; 
      };
    } else { // on device
#if defined(NDA_HAVE_CUTENSOR)
      cutensor::cutensor_desc<value_t,rank> a_t(a,oper);
      cutensor::cutensor_desc<value_t,rank> b_t(a,op::ID);
      std::string indx = default_index<uint8_t(rank)>();
      cutensor::permute(alpha, a_t, a.data(), indx, b_t, a.data(), indx);
#else
      static_assert(always_false<bool>," scale on device requires gpu tensor operations backend. ");
#endif
    }
  }

} // namespace nda::tensor
