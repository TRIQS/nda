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
   * Compute b(...) <- OP2 ( alpha * a(...),  beta * b(...)) using one of the tensor ops backend.
   *
   * @param b Out parameter. Can be a temporary view (hence the &&).
   * @param alpha/beta Scalars.
   * @param indxA Tensor index in einstein notation, provided as a string_view.
   * @param indxB Tensor index in einstein notation, provided as a string_view.
   * @param binary_oper Binary operation applied between a(...) and b(...).
   *
   * @Precondition :
   *       * Tensor ranks must match the size of the provided index list (string_view object).
   */
  template <Array X, Array Y>
  requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>)and(MemoryArray<Y> or nda::blas::is_conj_array_expr<Y>)
           and have_same_value_type_v<
              X, Y> and is_blas_lapack_v<get_value_t<X>> and is_blas_lapack_v<get_value_t<Y>>) void elementwise(get_value_t<X> const alpha,
                                                                                                                X const &x,
                                                                                                                std::string_view const indxX,
                                                                                                                get_value_t<X> const beta, Y &&y,
                                                                                                                std::string_view const indxY,
                                                                                                                op::TENSOR_OP binary_oper,
                                                                                                                devStream_t const stream = 0) {

    using nda::blas::is_conj_array_expr;
    using value_t = get_value_t<X>;
    auto to_mat   = []<typename Z>(Z &z) -> auto   &{
      if constexpr (is_conj_array_expr<Z>)
        return std::get<0>(z.a);
      else
        return z;
    };
    auto &a = to_mat(x);
    auto &b = to_mat(y);

    static constexpr bool conj_A = is_conj_array_expr<X>;
    static constexpr bool conj_B = is_conj_array_expr<Y>;

    using A = decltype(a);
    using B = decltype(b);
    static_assert(mem::have_compatible_addr_space<A, B>, "Matrices must have compatible memory address space");

    if (get_rank<A> != indxX.size()) NDA_RUNTIME_ERROR << "tensor::add: Rank mismatch \n";
    if (get_rank<B> != indxY.size()) NDA_RUNTIME_ERROR << "tensor::add: Rank mismatch \n";

    if constexpr (mem::have_device_compatible_addr_space<A, B>) {
#if defined(NDA_HAVE_CUTENSOR)
      op::TENSOR_OP a_op = conj_A ? op::CONJ : op::ID;
      op::TENSOR_OP b_op = conj_B ? op::CONJ : op::ID;
      cutensor::cutensor_desc<value_t, get_rank<A>> a_t(a);
      cutensor::cutensor_desc<value_t, get_rank<B>> b_t(b);
      cutensor::elementwise_binary(alpha, a_t, a_op, a.data(), indxX.data(), beta, b_t, b_op, b.data(), indxY.data(), b.data(), binary_oper, stream);
#else
      static_assert(always_false<bool>, " add on device requires gpu tensor operations backend. ");
#endif
    } else {
      // write routine that creates permuted view
      if (indxX != indxY) NDA_RUNTIME_ERROR << "tensor::assign: Index permutation not yet implemented.";
      if constexpr (get_rank<A> == get_rank<B>) {
        switch (binary_oper) {
          case op::SUM: b = alpha * a + beta * b; break;
          case op::MUL: b = (alpha * a) * (beta * b); break;
          default: NDA_RUNTIME_ERROR << "Unknown binary operation.";
        };
      }
    }
  }

  template <Array X, Array Y>
  requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>)and(MemoryArray<Y> or nda::blas::is_conj_array_expr<Y>)
           and have_same_value_type_v<
              X, Y> and is_blas_lapack_v<get_value_t<X>> and is_blas_lapack_v<get_value_t<Y>>) void elementwise(get_value_t<X> const alpha,
                                                                                                                X const &x, get_value_t<X> const beta,
                                                                                                                Y &&y, op::TENSOR_OP binary_oper,
                                                                                                                devStream_t const stream = 0) {
    std::string indxX = default_index<uint8_t(get_rank<X>)>();
    std::string indxY = default_index<uint8_t(get_rank<Y>)>();
    elementwise(alpha, x, indxX, beta, y, indxY, binary_oper, stream);
  }

  template <Array X, Array Y>
  requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>)and(MemoryArray<Y> or nda::blas::is_conj_array_expr<Y>)
           and have_same_value_type_v<
              X, Y> and is_blas_lapack_v<get_value_t<X>> and is_blas_lapack_v<get_value_t<Y>>) void elementwise(X const &x, Y &&y,
                                                                                                                op::TENSOR_OP binary_oper,
                                                                                                                devStream_t const stream = 0) {
    std::string indxX = default_index<uint8_t(get_rank<X>)>();
    std::string indxY = default_index<uint8_t(get_rank<Y>)>();
    elementwise(get_value_t<X>{1.0}, x, indxX, get_value_t<Y>{0.0}, y, indxY, binary_oper, stream);
  }

  template <Array X, Array Y>
  requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>)and(MemoryArray<Y> or nda::blas::is_conj_array_expr<Y>)
           and have_same_value_type_v<
              X, Y> and is_blas_lapack_v<get_value_t<X>> and is_blas_lapack_v<get_value_t<Y>>) void elementwise(X const &x, std::string indxX, Y &&y,
                                                                                                                std::string indxY,
                                                                                                                op::TENSOR_OP binary_oper,
                                                                                                                devStream_t const stream = 0) {
    elementwise(get_value_t<X>{1.0}, x, indxX, get_value_t<Y>{0.0}, y, indxY, binary_oper, stream);
  }

} // namespace nda::tensor
