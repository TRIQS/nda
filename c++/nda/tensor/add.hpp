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
#include "interface/cutensor_interface.h"
#endif

namespace nda::tensor {

  /**
   * Compute b(...) <- alpha a(...) + beta * b(...) using one of the tensor ops backend. 
   *
   * @param b Out parameter. Can be a temporary view (hence the &&).
   * @param ... Tensor index in einstein notation, provided as a string_view. 
   *
   * @Precondition :
   *       * Tensor ranks must match the size of the provided index list (string_view object).  
   */
  template <Array X, MemoryArray B>
  requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>) and                        
           have_same_value_type_v<X, B> and is_blas_lapack_v<get_value_t<X>>) 
  void add(get_value_t<X> alpha, X const &x, std::string_view const indxX, 
	   get_value_t<X> beta , B &&y, std::string_view const indxY) {

    using nda::blas::is_conj_array_expr;
    using value_t = get_value_t<X>;
    auto to_mat = []<typename Z>(Z const &z) -> auto & {
      if constexpr (is_conj_array_expr<Z>)
        return std::get<0>(z.a);
      else
        return z;
    };
    auto &a = to_mat(x);
    auto &b = to_mat(y);

    static constexpr bool conj_A = is_conj_array_expr<X>;

    // no conj in tblis yet!
    static_assert(not conj_A, "Error: No conj in tblis yet!");

    using A = decltype(a);
    static_assert(mem::have_same_addr_space_v<A, B>, "Matrices must have same memory address space");

    if( get_rank<A> != indxX.size() ) NDA_RUNTIME_ERROR <<"tensor::add: Rank mismatch \n";
    if( get_rank<B> != indxY.size() ) NDA_RUNTIME_ERROR <<"tensor::add: Rank mismatch \n";

    if constexpr (mem::on_host<A>) {
#if defined(NDA_HAVE_TBLIS)
      nda_tblis::tensor<value_t,get_rank<A>> a_t(a,alpha);
      nda_tblis::tensor<value_t,get_rank<B>> b_t(b,beta);
      ::tblis::tblis_tensor_add(NULL,NULL,&a_t,indxX.data(),&b_t,indxY.data());
#else
      static_assert(always_false<bool>," add on host requires cpu tensor operations backend. ");
#endif
    } else { // on device
#if defined(NDA_HAVE_CUTENSOR)
//      cutensor::termbyterm();
      static_assert(always_false<bool>," add on device cuTensor!!!. ");
#else
      static_assert(always_false<bool>," add on device requires gpu tensor operations backend. ");
#endif
    }
  }

} // namespace nda::tensor
