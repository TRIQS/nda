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
// Authors: Nils Wentzell

#pragma once
#include <complex>
#include "tools.hpp"
#include "interface/cxx_interface.hpp"
#include "../declarations.hpp"
#include "../mem/address_space.hpp"

namespace nda::blas {

  /**
   * Batched version of GEMM taking vectors of matrices as arguments
   */
  template <Matrix X, Matrix Y, MemoryMatrix C>
  requires((MemoryMatrix<X> or is_conj_array_expr<X>) and                        //
           (MemoryMatrix<Y> or is_conj_array_expr<Y>) and                        //
           have_same_value_type_v<X, Y, C> and is_blas_lapack_v<get_value_t<X>>) //
  void gemm_batch(get_value_t<X> alpha, std::vector<X> const &vx, std::vector<Y> const &vy, get_value_t<X> beta, std::vector<C> &vc) {

    EXPECTS(std::all_of(vx.begin(), vx.end(), [&vx](auto &x) { return x.shape() == vx[0].shape(); }));
    EXPECTS(std::all_of(vy.begin(), vy.end(), [&vy](auto &y) { return y.shape() == vy[0].shape(); }));
    EXPECTS(std::all_of(vc.begin(), vc.end(), [&vc](auto &c) { return c.shape() == vc[0].shape(); }));
    EXPECTS(vx.size() == vy.size() and vx.size() == vc.size());
    if (vx.empty()) return;
    int batch_count = vx.size();

    auto to_mat = []<typename Z>(Z & z) -> auto & {
      if constexpr (is_conj_array_expr<Z>)
        return std::get<0>(z.a);
      else
        return z;
    };
    auto &a0 = to_mat(vx[0]);
    auto &b0 = to_mat(vy[0]);
    auto &c0 = vc[0];

    static constexpr bool conj_A = is_conj_array_expr<X>;
    static constexpr bool conj_B = is_conj_array_expr<Y>;

    using A = decltype(a0);
    using B = decltype(b0);
    static_assert(mem::have_same_addr_space_v<A, B, C>, "Matrices must have same memory address space");

    EXPECTS(a0.extent(1) == b0.extent(0));
    EXPECTS(a0.extent(0) == c0.extent(0));
    EXPECTS(b0.extent(1) == c0.extent(1));

    // c is in C order: compute the transpose of the product in Fortran order
    if constexpr (has_C_layout<C>) {
      std::vector<std::decay_t<decltype(transpose(vx[0]))>> vxT;
      std::vector<std::decay_t<decltype(transpose(vy[0]))>> vyT;
      std::vector<std::decay_t<decltype(transpose(vc[0]))>> vcT;

      std::transform(vx.begin(), vx.end(), std::back_inserter(vxT), [](auto &x) { return transpose(x); });
      std::transform(vy.begin(), vy.end(), std::back_inserter(vyT), [](auto &y) { return transpose(y); });
      std::transform(vc.begin(), vc.end(), std::back_inserter(vcT), [](auto &c) { return transpose(c); });

      gemm_batch(alpha, vyT, vxT, beta, vcT);
      return;
    } else { // c is in Fortran order

      char op_a   = get_op<conj_A, /*transpose =*/has_C_layout<A>>;
      char op_b   = get_op<conj_B, /*transpose =*/has_C_layout<B>>;
      auto [m, k] = a0.shape();
      auto n      = b0.extent(1);

      auto to_ptr = [&to_mat]<typename Z>(Z & z) -> auto * {
        // Must be lapack compatible
        EXPECTS(to_mat(z).indexmap().min_stride() == 1);
        return to_mat(z).data();
      };

      vector<get_value_t<X> const *> a_ptrs(batch_count), b_ptrs(batch_count);
      std::transform(vx.begin(), vx.end(), a_ptrs.begin(), to_ptr);
      std::transform(vy.begin(), vy.end(), b_ptrs.begin(), to_ptr);

      vector<get_value_t<X> *> c_ptrs(batch_count);
      std::transform(vc.begin(), vc.end(), c_ptrs.begin(), to_ptr);

      if constexpr (mem::on_host<A>) {
        f77::gemm_batch(op_a, op_b, m, n, k, alpha, a_ptrs.data(), get_ld(a0), b_ptrs.data(), get_ld(b0), beta, c_ptrs.data(), get_ld(c0),
                        batch_count);
      } else { // on device
        cuda::gemm_batch(op_a, op_b, m, n, k, alpha, to_device(a_ptrs).data(), get_ld(a0), to_device(b_ptrs).data(), get_ld(b0), beta,
                         to_device(c_ptrs).data(), get_ld(c0), batch_count);
      }
    }
  }
  /**
   * Batched strided version of GEMM taking arrays of rank 3
   * as arguments, where the operation is performed for each
   * of the slices: c(i,_,_) = x(i,_,_) * y(i,_,_)
   */
  template <ArrayOfRank<3> X, ArrayOfRank<3> Y, MemoryArrayOfRank<3> C>
  requires((MemoryArrayOfRank<X, 3> or (is_conj_array_expr<X>)) and              //
           (MemoryArrayOfRank<Y, 3> or (is_conj_array_expr<Y>)) and              //
           have_same_value_type_v<X, Y, C> and is_blas_lapack_v<get_value_t<X>>) //
  void gemm_batch_strided(get_value_t<X> alpha, X const &x, Y const &y, get_value_t<X> beta, C &&c) {

    EXPECTS(x.shape()[0] == y.shape()[0] and y.shape()[0] == c.shape()[0]);
    auto to_arr = []<typename Z>(Z & z) -> auto & {
      if constexpr (is_conj_array_expr<Z>)
        return std::get<0>(z.a);
      else
        return z;
    };
    auto a = to_arr(x);
    auto b = to_arr(y);

    static constexpr bool conj_A = is_conj_array_expr<X>;
    static constexpr bool conj_B = is_conj_array_expr<Y>;

    using A = decltype(a);
    using B = decltype(b);
    static_assert(mem::have_same_addr_space_v<A, B, C>, "Arrays must have same memory address space");

    auto _  = nda::range::all;
    auto a0 = a(0, _, _);
    auto b0 = b(0, _, _);
    auto c0 = c(0, _, _);
    EXPECTS(a0.extent(1) == b0.extent(0));
    EXPECTS(a0.extent(0) == c0.extent(0));
    EXPECTS(b0.extent(1) == c0.extent(1));

    // Must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(b.indexmap().min_stride() == 1);
    EXPECTS(c.indexmap().min_stride() == 1);

    // c is in C order: compute the transpose of the product in Fortran order
    if constexpr (has_C_layout<C>) {
      gemm_batch_strided(alpha, transposed_view<1, 2>(y), transposed_view<1, 2>(x), beta, transposed_view<1, 2>(std::forward<C>(c)));
      return;
    } else { // c is in Fortran order
      char op_a   = get_op<conj_A, /*transpose =*/has_C_layout<A>>;
      char op_b   = get_op<conj_B, /*transpose =*/has_C_layout<B>>;
      auto [m, k] = a0.shape();
      auto n      = b0.extent(1);

      if constexpr (mem::on_host<A>) {
        f77::gemm_batch_strided(op_a, op_b, m, n, k, alpha, a.data(), get_ld(a0), a.strides()[0], b.data(), get_ld(b0), b.strides()[0], beta,
                                c.data(), get_ld(c0), c.strides()[0], a.extent(0));
      } else { // on device
        cuda::gemm_batch_strided(op_a, op_b, m, n, k, alpha, a.data(), get_ld(a0), a.indexmap().strides()[0], b.data(), get_ld(b0), b.strides()[0],
                                 beta, c.data(), get_ld(c0), c.indexmap().strides()[0], a.extent(0));
      }
    }
  }

} // namespace nda::blas
