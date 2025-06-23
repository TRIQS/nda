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
#include "nda/tensor/tools.hpp"
#include "nda/tensor/scale.hpp"

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
  requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>)and get_rank<X> == get_rank<B> and have_same_value_type_v<X, B>) void assign(
     get_value_t<X> alpha, X const &x, std::string indxA, B &&b, std::string indxB, [[maybe_unused]] devStream_t const stream = 0) {

    using nda::blas::is_conj_array_expr;
    using value_t      = get_value_t<X>;
    constexpr int rank = get_rank<X>;
    auto to_mat        = []<typename Z>(Z const &z) -> auto        &{
      if constexpr (is_conj_array_expr<Z>)
        return std::get<0>(z.a);
      else
        return z;
    };
    auto &a = to_mat(x);
    using A = decltype(a);

    static constexpr bool conj_A = is_conj_array_expr<X>;
    if constexpr (not is_blas_lapack_v<value_t>) {
      if (alpha != value_t{1} or conj_A) NDA_RUNTIME_ERROR << "tensor::assign: Integer type require alpha=1 and no conjugation.";
    }

    // prioritize device
    if constexpr (mem::have_device_compatible_addr_space<A, B>) { // on device
      auto rec_until = [](auto &&A_, auto &&B_) -> void {
        auto rec_until_impl = [](auto &&At, auto &&Bt, auto &impl) -> void {
          if constexpr ((rank == 1)
                        or ((has_layout_strided_1d<decltype(At)> or has_contiguous_layout<decltype(At)>)and(
                           has_layout_strided_1d<decltype(Bt)> or has_contiguous_layout<decltype(Bt)>))) {
            Bt() = At();
          } else {
            long n = Bt.extent(0);
            for (long i = 0; i < n; ++i) impl(At(i, ::nda::ellipsis{}), Bt(i, ::nda::ellipsis{}), impl);
          }
        };
        rec_until_impl(A_, B_, rec_until_impl);
      };

      if constexpr (is_blas_lapack_v<value_t>) {
#if defined(NDA_HAVE_CUTENSOR)
        cutensor::cutensor_desc<value_t, rank> a_t(a);
        cutensor::cutensor_desc<value_t, rank> b_t(b);
        op::TENSOR_OP oper = (conj_A ? op::CONJ : op::ID);
        cutensor::permute(alpha, a_t, oper, a.data(), indxA, b_t, b.data(), indxB, stream);
#else
        if (indxA != indxB) NDA_RUNTIME_ERROR << "tensor::assign: Index permutation not yet implemented in device memory without cutensor.";
        if (alpha == value_t{1} and not conj_A) {
          rec_until(a, b);
        } else {
          using Array_t = typename std::decay_t<A>::regular_type;
          Array_t a_copy(a);
          // this will currently fail at compile time without cutensor
          scale(alpha, a_copy, (conj_A ? (op::CONJ) : (op::ID)));
          rec_until(a_copy, b);
        }
#endif
      } else {
        rec_until(a, b);
      }
    } else if constexpr (mem::have_host_compatible_addr_space<A, B>) { // on host or unified
      // write routine that creates permuted view
      if (indxA != indxB) NDA_RUNTIME_ERROR << "tensor::assign: Index permutation not yet implemented in host memory.";
      if (conj_A) {
        b = alpha * nda::conj(a);
      } else {
        b = alpha * a;
      }
    } else {
      // fallback tp operator() with possible copy
      if (indxA != indxB) NDA_RUNTIME_ERROR << "tensor::assign: Index permutation not yet implemented with host/device case.";
      if (alpha == value_t{1} and not conj_A) {
        b() = a();
      } else {
        using Array_t = typename std::decay_t<A>::regular_type;
        Array_t a_copy(a);
        if constexpr (is_blas_lapack_v<value_t>) scale(alpha, a_copy, (conj_A ? (op::CONJ) : (op::ID)));
        b() = a_copy();
      }
    }
  }

  template <Array X, MemoryArray B>
  requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>)and get_rank<X> == get_rank<B> and have_same_value_type_v<X, B>) void assign(
     get_value_t<X> alpha, X const &x, B &&b, devStream_t const stream = 0) {
    constexpr int rank = get_rank<X>;
    std::string indx = default_index<uint8_t(rank)>();
    assign(alpha,x,indx,b,indx,stream);
  }

  template <Array X, MemoryArray B>
  requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>)and get_rank<X> == get_rank<B> and have_same_value_type_v<X, B>) void assign(
     X const &x, B &&b, devStream_t const stream = 0) {
    constexpr int rank = get_rank<X>;
    std::string indx = default_index<uint8_t(rank)>();
    assign(get_value_t<X>{1}, x, indx, b, indx, stream);
  }

  template <Array X, MemoryArray B>
  requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>)and get_rank<X> == get_rank<B> and have_same_value_type_v<X, B>) void assign(
     X const &x, std::string indxA, B &&b, std::string indxB, devStream_t const stream = 0) {
    assign(get_value_t<X>{1}, x, indxA, b, indxB, stream);
  }

} // namespace nda::tensor
