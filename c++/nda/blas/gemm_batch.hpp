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
#include "../basic_array.hpp"
#include "../basic_functions.hpp"
#include "../mem/address_space.hpp"

namespace nda::blas {

  /**
   * Batched version of GEMM taking vectors of matrices as arguments
   */
  template <bool VBATCH = false, Matrix X, Matrix Y, MemoryMatrix C>
  requires((MemoryMatrix<X> or is_conj_array_expr<X>) and                        //
           (MemoryMatrix<Y> or is_conj_array_expr<Y>) and                        //
           have_same_value_type_v<X, Y, C> and is_blas_lapack_v<get_value_t<X>>) //
  void gemm_batch(get_value_t<X> alpha, std::vector<X> const &vx, std::vector<Y> const &vy, get_value_t<X> beta, std::vector<C> &vc) {

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

    // c is in C order: compute the transpose of the product in Fortran order
    if constexpr (has_C_layout<C>) {

      auto map_transpose = [](auto &v) {
        auto vT = std::vector<std::decay_t<decltype(transpose(v[0]))>>{};
        vT.reserve(v.size());
        std::transform(v.begin(), v.end(), std::back_inserter(vT), [](auto &x) { return transpose(x); });
        return vT;
      };
      auto vcT = map_transpose(vc);
      gemm_batch<VBATCH>(alpha, map_transpose(vy), map_transpose(vx), beta, vcT);
      return;
    } else { // c is in Fortran order

      // For operations on the device, use unified memory for vector of ints or ptrs
      auto constexpr vec_adr_spc = []() { return mem::on_host<C> ? mem::Host : mem::Unified; }();

      // Convert the vector of matrices into the associated vector of pointers
      auto get_ptrs = [&to_mat]<typename V>(V &v) {
        EXPECTS(std::all_of(v.begin(), v.end(),
                            [&v, &to_mat](auto &z) { return (VBATCH or z.shape() == v[0].shape()) and to_mat(z).indexmap().min_stride() == 1; }));
        using value_t = get_value_t<typename V::value_type>;
        using ptr_t   = std::conditional_t<std::is_const_v<V>, value_t const *, value_t *>;
        auto v_ptrs   = vector<ptr_t, heap<vec_adr_spc>>(v.size());
        std::transform(v.begin(), v.end(), v_ptrs.begin(), [&to_mat](auto &z) { return to_mat(z).data(); });
        return v_ptrs;
      };
      auto a_ptrs = get_ptrs(vx);
      auto b_ptrs = get_ptrs(vy);
      auto c_ptrs = get_ptrs(vc);

      char op_a = get_op<conj_A, /*transpose =*/has_C_layout<A>>;
      char op_b = get_op<conj_B, /*transpose =*/has_C_layout<B>>;

      if constexpr (VBATCH) {

        // Create vectors of size 'batch_count + 1' as required by Magma
        vector<int, heap<vec_adr_spc>> vm(batch_count + 1), vk(batch_count + 1), vn(batch_count + 1), vlda(batch_count + 1), vldb(batch_count + 1),
           vldc(batch_count + 1);

        for (auto i : range(batch_count)) {
          auto &ai = to_mat(vx[i]);
          auto &bi = to_mat(vy[i]);
          auto &ci = vc[i];

          EXPECTS(ai.extent(1) == bi.extent(0));
          EXPECTS(ai.extent(0) == ci.extent(0));
          EXPECTS(bi.extent(1) == ci.extent(1));

          vm[i] = ai.extent(0);
          vk[i] = ai.extent(1);
          vn[i] = bi.extent(1);

          vlda[i] = get_ld(ai);
          vldb[i] = get_ld(bi);
          vldc[i] = get_ld(ci);
        }

        if constexpr (mem::on_host<A>) {
          f77::gemm_vbatch(op_a, op_b, vm.data(), vn.data(), vk.data(), alpha, a_ptrs.data(), vlda.data(), b_ptrs.data(), vldb.data(), beta,
                           c_ptrs.data(), vldc.data(), batch_count);
        } else { // on device
#if defined(NDA_HAVE_DEVICE)
          device::gemm_vbatch(op_a, op_b, vm.data(), vn.data(), vk.data(), alpha, a_ptrs.data(), vlda.data(), b_ptrs.data(), vldb.data(), beta,
                            c_ptrs.data(), vldc.data(), batch_count);
#else
          static_assert(always_false<bool>," blas on device without gpu support! Compile for GPU. ");
#endif
        }
      } else {

        EXPECTS(a0.extent(1) == b0.extent(0));
        EXPECTS(a0.extent(0) == c0.extent(0));
        EXPECTS(b0.extent(1) == c0.extent(1));

        auto [m, k] = a0.shape();
        auto n      = b0.extent(1);

        if constexpr (mem::on_host<A>) {
          f77::gemm_batch(op_a, op_b, m, n, k, alpha, a_ptrs.data(), get_ld(a0), b_ptrs.data(), get_ld(b0), beta, c_ptrs.data(), get_ld(c0),
                          batch_count);
        } else { // on device
#if defined(NDA_HAVE_DEVICE)
          device::gemm_batch(op_a, op_b, m, n, k, alpha, a_ptrs.data(), get_ld(a0), b_ptrs.data(), get_ld(b0), beta, c_ptrs.data(), get_ld(c0),
                           batch_count);
#else
          static_assert(always_false<bool>," blas on device without gpu support! Compile for GPU. ");
#endif
        }
      }
    }
  }

  /**
   * VBatched version of GEMM taking vectors of matrices as arguments
   */
  template <Matrix X, Matrix Y, MemoryMatrix C>
  void gemm_vbatch(get_value_t<X> alpha, std::vector<X> const &vx, std::vector<Y> const &vy, get_value_t<X> beta, std::vector<C> &vc) {
    gemm_batch</*VBATCH=*/true>(alpha, vx, vy, beta, vc);
  }

  /**
   * Batched strided version of GEMM taking arrays of rank 3
   * as arguments, where the operation is performed for each
   * of the slices:
   *
   *   c(i,_,_) = x(i,_,_) * y(i,_,_)
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
      //Reconsider ..
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
#if defined(NDA_HAVE_DEVICE)
        device::gemm_batch_strided(op_a, op_b, m, n, k, alpha, a.data(), get_ld(a0), a.indexmap().strides()[0], b.data(), get_ld(b0), b.strides()[0],
                                 beta, c.data(), get_ld(c0), c.indexmap().strides()[0], a.extent(0));
#else
        static_assert(always_false<bool>," blas on device without gpu support! Compile for GPU. ");
#endif
      }
    }
  }

} // namespace nda::blas
