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

#ifndef NDA_HAVE_DEVICE
#include "../device.hpp"
#endif

#if defined(NDA_HAVE_CUTENSOR)
#include "interface/cutensor_interface.hpp"
#endif

namespace nda::tensor {

  /**
   * Compute d(...) <- ( OP_ABC ( OP_AB( alpha * a(...),  beta * b(...)), gamma * C()...)) using one of the tensor ops backend.
   *
   * @param d Out parameter. Can be a temporary view (hence the &&).
   * @param alpha/beta/gamma Scalars.
   * @param indxA  Tensor index in einstein notation, provided as a string_view.
   * @param indxB  Tensor index in einstein notation, provided as a string_view.
   * @param indxC  Tensor index in einstein notation, provided as a string_view.
   * @param operAB  Binary operation applied between a(...) and b(...).
   * @param operABC Binary operation applied between operAB(a(...),b(...)) and c(...).
   *
   * @Precondition :
   *       * Tensor ranks must match the size of the provided index list (string_view object).
   */
  template <Array X, Array Y, Array Z>
    requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>) and (MemoryArray<Y> or nda::blas::is_conj_array_expr<Y>) and ((MemoryArray<Z> or nda::blas::is_conj_array_expr<Z>))
             and have_same_value_type_v<X, Y, Z> and is_blas_lapack_v<get_value_t<X>> and is_blas_lapack_v<get_value_t<Y>> and is_blas_lapack_v<get_value_t<Z>>)
  void elementwise_trinary(get_value_t<X> const alpha, X const &x, std::string_view const indxX, 
                   get_value_t<X> const  beta, Y const &y, std::string_view const indxY, 
                   get_value_t<X> const gamma, Z &&z, std::string_view const indxZ, 
                   op::TENSOR_OP operAB, op::TENSOR_OP operABC, [[maybe_unused]] devStream_t const stream = 0) {

    using nda::blas::is_conj_array_expr;
    auto to_mat = []<typename W>(W &w) -> auto & {
      if constexpr (is_conj_array_expr<W>)
        return std::get<0>(w.a);
      else
        return w;
    };
    auto &a = to_mat(x);
    auto &b = to_mat(y);
    auto &c = to_mat(z);

    using A = decltype(a);
    using B = decltype(b);
    using C = decltype(c);
    static_assert(mem::have_compatible_addr_space<A, B, C>, "Matrices must have compatible memory address space");

    if (get_rank<A> != indxX.size()) NDA_RUNTIME_ERROR << "tensor::add: Rank mismatch \n";
    if (get_rank<B> != indxY.size()) NDA_RUNTIME_ERROR << "tensor::add: Rank mismatch \n";
    if (get_rank<C> != indxZ.size()) NDA_RUNTIME_ERROR << "tensor::add: Rank mismatch \n";

    if constexpr (mem::have_device_compatible_addr_space<A, B>) {
#if defined(NDA_HAVE_CUTENSOR)
      using value_t                = get_value_t<X>;
      static constexpr bool conj_A = is_conj_array_expr<X>;
      static constexpr bool conj_B = is_conj_array_expr<Y>;
      static constexpr bool conj_C = is_conj_array_expr<Z>;
      op::TENSOR_OP a_op           = conj_A ? op::CONJ : op::ID;
      op::TENSOR_OP b_op           = conj_B ? op::CONJ : op::ID;
      op::TENSOR_OP c_op           = conj_C ? op::CONJ : op::ID;
      cutensor::cutensor_desc<value_t, get_rank<A>> a_t(a);
      cutensor::cutensor_desc<value_t, get_rank<B>> b_t(b);
      cutensor::cutensor_desc<value_t, get_rank<C>> c_t(c);
      cutensor::elementwise_trinary(alpha, a_t, a_op, a.data(), indxX.data(), beta, b_t, b_op, b.data(), indxY.data(), gamma, c_t, c_op, c.data(), indxZ.data(), c.data(), operAB, operABC, stream);
#else
      compile_error_no_gpu();
#endif
    } else {
      if (indxX != indxY) NDA_RUNTIME_ERROR << "tensor::assign: Index permutation not yet implemented.";
      if (indxX != indxZ) NDA_RUNTIME_ERROR << "tensor::assign: Index permutation not yet implemented.";
      if constexpr ((get_rank<A> == get_rank<B>) and (get_rank<A> == get_rank<C>)) {
        C tmp = a;
        switch (operAB) {
          case op::SUM: tmp = alpha * tmp + beta * b; break;
          case op::MUL: tmp = (alpha * tmp) * (beta * b); break;
          default: NDA_RUNTIME_ERROR << "Unknown binary operation.";
        };
        switch (operABC) {
          case op::SUM: c = tmp + gamma * c; break;
          case op::MUL: c = tmp * (gamma * c); break;
          default: NDA_RUNTIME_ERROR << "Unknown binary operation.";
        };
      }
    }
  }
/*
  template <Array X, Array Y>
    requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>) and (MemoryArray<Y> or nda::blas::is_conj_array_expr<Y>)
             and have_same_value_type_v<X, Y> and is_blas_lapack_v<get_value_t<X>> and is_blas_lapack_v<get_value_t<Y>>)
  void elementwise(get_value_t<X> const alpha, X const &x, get_value_t<X> const beta, Y &&y, op::TENSOR_OP binary_oper,
                   devStream_t const stream = 0) {
    std::string indxX = default_index<uint8_t(get_rank<X>)>();
    std::string indxY = default_index<uint8_t(get_rank<Y>)>();
    elementwise(alpha, x, indxX, beta, y, indxY, binary_oper, stream);
  }

  template <Array X, Array Y>
    requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>) and (MemoryArray<Y> or nda::blas::is_conj_array_expr<Y>)
             and have_same_value_type_v<X, Y> and is_blas_lapack_v<get_value_t<X>> and is_blas_lapack_v<get_value_t<Y>>)
  void elementwise(X const &x, Y &&y, op::TENSOR_OP binary_oper, devStream_t const stream = 0) {
    std::string indxX = default_index<uint8_t(get_rank<X>)>();
    std::string indxY = default_index<uint8_t(get_rank<Y>)>();
    elementwise(get_value_t<X>{1.0}, x, indxX, get_value_t<Y>{0.0}, y, indxY, binary_oper, stream);
  }

  template <Array X, Array Y>
    requires((MemoryArray<X> or nda::blas::is_conj_array_expr<X>) and (MemoryArray<Y> or nda::blas::is_conj_array_expr<Y>)
             and have_same_value_type_v<X, Y> and is_blas_lapack_v<get_value_t<X>> and is_blas_lapack_v<get_value_t<Y>>)
  void elementwise(X const &x, std::string indxX, Y &&y, std::string indxY, op::TENSOR_OP binary_oper, devStream_t const stream = 0) {
    elementwise(get_value_t<X>{1.0}, x, indxX, get_value_t<Y>{0.0}, y, indxY, binary_oper, stream);
  }
*/

} // namespace nda::tensor
