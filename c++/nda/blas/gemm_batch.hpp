// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to batched versions of the BLAS `gemm` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "./tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../layout_transforms.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

#ifndef NDA_HAVE_DEVICE
#include "../device.hpp"
#endif

#include <algorithm>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <vector>

namespace nda::blas {

  /**
   * @addtogroup linalg_blas
   * @{
   */

  /**
   * @brief Implements a batched version of nda::blas::gemm taking vectors of matrices as arguments.
   *
   * @details This routine is a batched version of nda::blas::gemm, performing multiple `gemm` operations in a single
   * call. Each `gemm` operation performs a matrix-matrix product with general matrices.
   *
   * @tparam VBATCH Allow for variable sized matrices.
   * @tparam A nda::Matrix type.
   * @tparam B nda::Matrix type.
   * @tparam C nda::MemoryMatrix type.
   * @param alpha Input scalar.
   * @param va std::vector of input matrices.
   * @param vb std::vector of input matrices.
   * @param beta Input scalar.
   * @param vc std::vector of input/output matrices.
   */
  template <bool VBATCH = false, Matrix A, Matrix B, MemoryMatrix C>
    requires((MemoryMatrix<A> or is_conj_array_expr<A>) and (MemoryMatrix<B> or is_conj_array_expr<B>)
             and have_same_value_type_v<A, B, C> and is_blas_lapack_v<get_value_t<A>>)
  void gemm_batch(get_value_t<A> alpha, std::vector<A> const &va, std::vector<B> const &vb, get_value_t<A> beta, std::vector<C> &vc) {
    // check sizes
    EXPECTS(va.size() == vb.size() and va.size() == vc.size());
    if (va.empty()) return;
    int batch_count = va.size();

    // get underlying matrix in case it is given as a lazy expression
    auto to_mat = []<typename Z>(Z &z) -> auto & {
      if constexpr (is_conj_array_expr<Z>)
        return std::get<0>(z.a);
      else
        return z;
    };
    auto &a0 = to_mat(va[0]);
    auto &b0 = to_mat(vb[0]);
    auto &c0 = vc[0];

    // compile-time checks
    using mat_a_type = decltype(a0);
    using mat_b_type = decltype(b0);
    static_assert(mem::have_compatible_addr_space<mat_a_type, mat_b_type, C>, "Error in nda::blas::gemm_batch: Incompatible memory address spaces");

    // c is in C order: compute the transpose of the product in Fortran order
    if constexpr (has_C_layout<C>) {
      // transpose each matrix in the given vector
      auto map_transpose = [](auto &v) {
        auto vt = std::vector<std::decay_t<decltype(transpose(v[0]))>>{};
        vt.reserve(v.size());
        std::transform(v.begin(), v.end(), std::back_inserter(vt), [](auto &x) { return transpose(x); });
        return vt;
      };
      auto vct = map_transpose(vc);
      gemm_batch<VBATCH>(alpha, map_transpose(vb), map_transpose(va), beta, vct);
      return;
    } else { // c is in Fortran order
      // for operations on the device, use unified memory for vector of ints or ptrs
      auto constexpr vec_adr_spc = []() { return mem::on_host<C> ? mem::Host : mem::Unified; }();

      // convert the vector of matrices into the associated vector of pointers
      auto get_ptrs = [&to_mat]<typename V>(V &v) {
        EXPECTS(std::all_of(v.begin(), v.end(),
                            [&v, &to_mat](auto &z) { return (VBATCH or z.shape() == v[0].shape()) and to_mat(z).indexmap().min_stride() == 1; }));
        using value_t = get_value_t<typename V::value_type>;
        using ptr_t   = std::conditional_t<std::is_const_v<V>, value_t const *, value_t *>;
        auto v_ptrs   = nda::vector<ptr_t, heap<vec_adr_spc>>(v.size());
        std::transform(v.begin(), v.end(), v_ptrs.begin(), [&to_mat](auto &z) { return to_mat(z).data(); });
        return v_ptrs;
      };
      auto a_ptrs = get_ptrs(va);
      auto b_ptrs = get_ptrs(vb);
      auto c_ptrs = get_ptrs(vc);

      // matrices have different sizes
      if constexpr (VBATCH) {
        // create vectors of size 'batch_count + 1' as required by Magma
        nda::vector<int, heap<vec_adr_spc>> vm(batch_count + 1), vk(batch_count + 1), vn(batch_count + 1), vlda(batch_count + 1),
           vldb(batch_count + 1), vldc(batch_count + 1);

        for (auto i : range(batch_count)) {
          auto &ai = to_mat(va[i]);
          auto &bi = to_mat(vb[i]);
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

        if constexpr (mem::have_device_compatible_addr_space<mat_a_type, mat_b_type, C>) {
#if defined(NDA_HAVE_DEVICE)
          device::gemm_vbatch(get_op<A>, get_op<B>, vm.data(), vn.data(), vk.data(), alpha, a_ptrs.data(), vlda.data(), b_ptrs.data(), vldb.data(),
                              beta, c_ptrs.data(), vldc.data(), batch_count);
#else
          compile_error_no_gpu();
#endif
        } else {
          f77::gemm_vbatch(get_op<A>, get_op<B>, vm.data(), vn.data(), vk.data(), alpha, a_ptrs.data(), vlda.data(), b_ptrs.data(), vldb.data(), beta,
                           c_ptrs.data(), vldc.data(), batch_count);
        }
      } else {
        // all matrices have the same size
        EXPECTS(a0.extent(1) == b0.extent(0));
        EXPECTS(a0.extent(0) == c0.extent(0));
        EXPECTS(b0.extent(1) == c0.extent(1));

        auto [m, k] = a0.shape();
        auto n      = b0.extent(1);

        if constexpr (mem::have_device_compatible_addr_space<mat_a_type, mat_b_type, C>) {
#if defined(NDA_HAVE_DEVICE)
          device::gemm_batch(get_op<A>, get_op<B>, m, n, k, alpha, a_ptrs.data(), get_ld(a0), b_ptrs.data(), get_ld(b0), beta, c_ptrs.data(),
                             get_ld(c0), batch_count);
#else
          compile_error_no_gpu();
#endif
        } else {
          f77::gemm_batch(get_op<A>, get_op<B>, m, n, k, alpha, a_ptrs.data(), get_ld(a0), b_ptrs.data(), get_ld(b0), beta, c_ptrs.data(), get_ld(c0),
                          batch_count);
        }
      }
    }
  }

  /**
   * @brief Wrapper of nda::blas::gemm_batch that allows variable sized matrices.
   *
   * @tparam A nda::Matrix type.
   * @tparam B nda::Matrix type.
   * @tparam C nda::MemoryMatrix type.
   * @param alpha Input scalar.
   * @param va std::vector of input matrices.
   * @param vb std::vector of input matrices.
   * @param beta Input scalar.
   * @param vc std::vector of input/output matrices.
   */
  template <Matrix A, Matrix B, MemoryMatrix C>
  void gemm_vbatch(get_value_t<A> alpha, std::vector<A> const &va, std::vector<B> const &vb, get_value_t<A> beta, std::vector<C> &vc) {
    gemm_batch</* VBATCH = */ true>(alpha, va, vb, beta, vc);
  }

  /**
   * @brief Implements a strided batched version of nda::blas::gemm taking 3-dimensional arrays as arguments.
   *
   * @details This function is similar to nda::blas::gemm_batch except that it takes 3-dimensional arrays as arguments
   * instead of vectors of matrices. The first dimension of the arrays indexes the matrices to be multiplied.
   *
   * @tparam A nda::ArrayOfRank<3> type.
   * @tparam B nda::ArrayOfRank<3> type.
   * @tparam C nda::ArrayOfRank<3> type.
   * @param alpha Input scalar.
   * @param a 3-dimensional input array.
   * @param b 3-dimensional input array.
   * @param beta Input scalar.
   * @param c 3-dimensional input/output array.
   */
  template <ArrayOfRank<3> A, ArrayOfRank<3> B, MemoryArrayOfRank<3> C>
    requires((MemoryArrayOfRank<A, 3> or (is_conj_array_expr<A>)) and (MemoryArrayOfRank<B, 3> or (is_conj_array_expr<B>))
             and have_same_value_type_v<A, B, C> and is_blas_lapack_v<get_value_t<A>>)
  void gemm_batch_strided(get_value_t<A> alpha, A const &a, B const &b, get_value_t<A> beta, C &&c) {
    // check number of matrices
    EXPECTS(a.shape()[0] == b.shape()[0] and a.shape()[0] == c.shape()[0]);

    // get underlying array in case it is given as a lazy expression
    auto to_arr = []<typename Z>(Z &z) -> auto & {
      if constexpr (is_conj_array_expr<Z>)
        return std::get<0>(z.a);
      else
        return z;
    };
    auto arr_a = to_arr(a);
    auto arr_b = to_arr(b);

    // compile-time check
    using arr_a_type = decltype(arr_a);
    using arr_b_type = decltype(arr_b);
    static_assert(mem::have_compatible_addr_space<arr_a_type, arr_b_type, C>,
                  "Error in nda::blas::gemm_batch_strided: Incompatible memory address spaces");

    // runtime checks
    auto _  = nda::range::all;
    auto a0 = arr_a(0, _, _);
    auto b0 = arr_b(0, _, _);
    auto c0 = c(0, _, _);
    EXPECTS(a0.extent(1) == b0.extent(0));
    EXPECTS(a0.extent(0) == c0.extent(0));
    EXPECTS(b0.extent(1) == c0.extent(1));
    EXPECTS(arr_a.indexmap().min_stride() == 1);
    EXPECTS(arr_b.indexmap().min_stride() == 1);
    EXPECTS(c.indexmap().min_stride() == 1);

    // c is in C order: compute the transpose of the product in Fortran order
    if constexpr (has_C_layout<C>) {
      gemm_batch_strided(alpha, transposed_view<1, 2>(b), transposed_view<1, 2>(a), beta, transposed_view<1, 2>(std::forward<C>(c)));
      return;
    } else { // c is in Fortran order
      auto [m, k] = a0.shape();
      auto n      = b0.extent(1);

      if constexpr (mem::have_device_compatible_addr_space<arr_a_type, arr_b_type, C>) {
#if defined(NDA_HAVE_DEVICE)
        device::gemm_batch_strided(get_op<A>, get_op<B>, m, n, k, alpha, arr_a.data(), get_ld(a0), arr_a.strides()[0], arr_b.data(), get_ld(b0),
                                   arr_b.strides()[0], beta, c.data(), get_ld(c0), c.strides()[0], arr_a.extent(0));
#else
        compile_error_no_gpu();
#endif
      } else {
        f77::gemm_batch_strided(get_op<A>, get_op<B>, m, n, k, alpha, arr_a.data(), get_ld(a0), arr_a.strides()[0], arr_b.data(), get_ld(b0),
                                arr_b.strides()[0], beta, c.data(), get_ld(c0), c.strides()[0], arr_a.extent(0));
      }
    }
  }

  /** @} */

} // namespace nda::blas
