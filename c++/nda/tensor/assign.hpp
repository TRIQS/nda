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

 /*
  * B(...) = a*Op(A(...))
  */ 
  template <Array X, MemoryArray B>
    requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>) and
             get_rank<X> == get_rank<B> and
             is_blas_lapack_v<get_value_t<X>> and is_blas_lapack_v<get_value_t<B>> and 
             have_same_value_type_v<X, B>)
  void assign(get_value_t<X> alpha, X const& x, std::string indxA,
                                    B &&     b, std::string indxB) {

    using nda::blas::is_conj_array_expr;
    using value_t = get_value_t<X>;
    constexpr int rank = get_rank<X>;
    auto to_mat   = []<typename Z>(Z const &z) -> auto   &{
      if constexpr (is_conj_array_expr<Z>)
        return std::get<0>(z.a);
      else
        return z;
    };
    auto &a = to_mat(x);

    static constexpr bool conj_A = is_conj_array_expr<X>;

    if constexpr (mem::on_host<X>) {
      // write routine that creates permuted view
      if(indxA != indxB)
        NDA_RUNTIME_ERROR << "tensor::assign: Index permutation not yet implemented.";
      if(conj_A) {
        b = alpha * nda::conj( a );
      } else {
        b = alpha * a;
      }  
    } else { // on device
#if defined(NDA_HAVE_CUTENSOR)
      cutensor::cutensor_desc<value_t, rank> a_t(a);
      cutensor::cutensor_desc<value_t, rank> b_t(b);
      op::TENSOR_OP oper = ( conj_A ? op::CONJ : op::ID );
      cutensor::permute(alpha, a_t, oper, a.data(), indxA, b_t, b.data(), indxB);
#else
      static_assert(always_false<bool>, " scale on device requires gpu tensor operations backend. ");
#endif
    }
  }

  template <Array X, MemoryArray B>
    requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>) and
             get_rank<X> == get_rank<B> and
             is_blas_lapack_v<get_value_t<X>> and is_blas_lapack_v<get_value_t<B>> and 
             have_same_value_type_v<X, B>)
  void assign(get_value_t<X> alpha, X const& x, B &&b)
  {
    constexpr int rank = get_rank<X>;
    std::string indx = default_index<uint8_t(rank)>();
    assign(alpha,x,indx,b,indx);
  }
  
  template <Array X, MemoryArray B>
    requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>) and
             get_rank<X> == get_rank<B> and
             is_blas_lapack_v<get_value_t<X>> and is_blas_lapack_v<get_value_t<B>> and 
             have_same_value_type_v<X, B>)
  void assign(X const& x, B &&b)
  {
    constexpr int rank = get_rank<X>;
    std::string indx = default_index<uint8_t(rank)>();
    assign(get_value_t<X>{1.0},x,indx,b,indx);
  }

  template <Array X, MemoryArray B>
    requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>) and
             get_rank<X> == get_rank<B> and
             is_blas_lapack_v<get_value_t<X>> and is_blas_lapack_v<get_value_t<B>> and 
             have_same_value_type_v<X, B>)
  void assign(X const& x, std::string indxA, B &&b, std::string indxB) 
  {
    assign(get_value_t<X>{1.0},x,indxA,b,indxB);
  } 


} // namespace nda::tensor
