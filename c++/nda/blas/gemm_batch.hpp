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
#endif // NDA_HAVE_DEVICE

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

  namespace detail {

    // Get a vector of transpose matrices from a given vector of matrices.
    auto get_transpose_vector(auto &&v) {
      auto v_t = std::vector<std::decay_t<decltype(transpose(v[0]))>>{};
      v_t.reserve(v.size());
      std::transform(v.begin(), v.end(), std::back_inserter(v_t), [](auto &x) { return transpose(x); });
      return v_t;
    }

    // Get a vector of pointers to the memory of matrices from a given vector of matrices.
    template <typename T, bool is_vbatch, nda::mem::AddressSpace vec_addr_spc>
    auto get_ptr_vector(auto &&v) {
      EXPECTS(std::ranges::all_of(v, [&v](auto &A) { return is_vbatch or A.shape() == v[0].shape(); }));
      EXPECTS(std::ranges::all_of(v, [](auto &A) { return get_array(A).indexmap().min_stride() == 1; }));
      auto v_ptrs = nda::vector<T, heap<vec_addr_spc>>(v.size());
      std::transform(v.begin(), v.end(), v_ptrs.begin(), [](auto &z) { return get_array(z).data(); });
      return v_ptrs;
    }

  } // namespace detail

  /**
   * @brief Interface to MKL's/CUDA's `gemm_batch` and `gemm_vbatch` routines.
   *
   * @details This routine is a batched version of nda::blas::gemm, performing multiple `gemm` operations in a single
   * call. Each `gemm` operation performs a matrix-matrix produc.
   *
   * If `is_vbatch` is true, the matrices are allowed to have different sizes. Otherwise, they are required to have the
   * same size.
   *
   * See also nda::blas::gemm for more details.
   *
   * @tparam is_vbatch Allow variable sized matrices.
   * @tparam A nda::Matrix type.
   * @tparam B nda::Matrix type.
   * @tparam C nda::MemoryMatrix type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param va `std::vector` of input matrices.
   * @param vb `std::vector` of input matrices.
   * @param beta Input scalar \f$ \beta \f$.
   * @param vc `std::vector` of input/output matrices.
   */
  template <bool is_vbatch = false, Matrix A, Matrix B, MemoryMatrix C>
    requires((MemoryMatrix<A> or is_conj_array_expr<A>) and (MemoryMatrix<B> or is_conj_array_expr<B>)
             and have_same_value_type_v<A, B, C> and mem::have_compatible_addr_space<A, B, C> and is_blas_lapack_v<get_value_t<A>>)
  void gemm_batch(get_value_t<A> alpha, std::vector<A> const &va, std::vector<B> const &vb, get_value_t<A> beta, std::vector<C> &vc) {
    // check sizes of input vectors and return if they are empty
    EXPECTS(va.size() == vb.size() and va.size() == vc.size());
    if (va.empty()) return;
    auto const batch_count = va.size();

    // if C is in C-layout, compute the transpose of the product in Fortran order
    if constexpr (has_C_layout<C>) {
      auto vc_t = detail::get_transpose_vector(vc);
      return gemm_batch<is_vbatch>(alpha, detail::get_transpose_vector(vb), detail::get_transpose_vector(va), beta, vc_t);
    } else {
      // for operations on the device, use unified memory for vector of ints or ptrs
      auto constexpr vec_addr_spc = []() { return mem::on_host<C> ? mem::Host : mem::Unified; }();

      // convert the vector of matrices to the corresponding vector of pointers
      auto a_ptrs = detail::get_ptr_vector<get_value_t<decltype(va[0])> const *, is_vbatch, vec_addr_spc>(va);
      auto b_ptrs = detail::get_ptr_vector<get_value_t<decltype(vb[0])> const *, is_vbatch, vec_addr_spc>(vb);
      auto c_ptrs = detail::get_ptr_vector<get_value_t<decltype(vc[0])> *, is_vbatch, vec_addr_spc>(vc);

      // check for conjugate lazy expressions and C-layouts
      char op_a = get_op<is_conj_array_expr<A>, has_C_layout<A>>;
      char op_b = get_op<is_conj_array_expr<B>, has_C_layout<B>>;

      // either call gemm_vbatch or gemm_batch
      if constexpr (is_vbatch) {
        // create vectors to store shapes and leading dimensions of size 'batch_count + 1' as required by Magma
        nda::vector<int, heap<vec_addr_spc>> vm(batch_count + 1), vk(batch_count + 1), vn(batch_count + 1), vlda(batch_count + 1),
           vldb(batch_count + 1), vldc(batch_count + 1);

        for (auto i : range(batch_count)) {
          auto &&mat_a = get_array(va[i]);
          auto &&mat_b = get_array(vb[i]);
          auto &&mat_c = get_array(vc[i]);

          // check the dimensions of the input/output arrays/views
          auto const [m, k] = mat_a.shape();
          auto const [l, n] = mat_b.shape();
          EXPECTS(k == l);
          EXPECTS(m == mat_c.extent(0));
          EXPECTS(n == mat_c.extent(1));

          // store shapes and leading dimensions
          vm[i]   = m;
          vk[i]   = k;
          vn[i]   = n;
          vlda[i] = get_ld(mat_a);
          vldb[i] = get_ld(mat_b);
          vldc[i] = get_ld(mat_c);
        }

        // perform the actual library call
        if constexpr (mem::have_device_compatible_addr_space<A, B, C>) {
#if defined(NDA_HAVE_DEVICE)
          device::gemm_vbatch(op_a, op_b, vm.data(), vn.data(), vk.data(), alpha, a_ptrs.data(), vlda.data(), b_ptrs.data(), vldb.data(), beta,
                              c_ptrs.data(), vldc.data(), batch_count);
#else
          compile_error_no_gpu();
#endif
        } else {
          f77::gemm_vbatch(op_a, op_b, vm.data(), vn.data(), vk.data(), alpha, a_ptrs.data(), vlda.data(), b_ptrs.data(), vldb.data(), beta,
                           c_ptrs.data(), vldc.data(), batch_count);
        }
      } else {
        auto &&mat_a = get_array(va[0]);
        auto &&mat_b = get_array(vb[0]);
        auto &&mat_c = get_array(vc[0]);

        // check the dimensions of the input/output arrays/views
        auto const [m, k] = mat_a.shape();
        auto const [l, n] = mat_b.shape();
        EXPECTS(k == l);
        EXPECTS(m == mat_c.extent(0));
        EXPECTS(n == mat_c.extent(1));

        // perform the actual library call
        if constexpr (mem::have_device_compatible_addr_space<A, B, C>) {
#if defined(NDA_HAVE_DEVICE)
          device::gemm_batch(op_a, op_b, m, n, k, alpha, a_ptrs.data(), get_ld(mat_a), b_ptrs.data(), get_ld(mat_b), beta, c_ptrs.data(),
                             get_ld(mat_c), batch_count);
#else
          compile_error_no_gpu();
#endif
        } else {
          f77::gemm_batch(op_a, op_b, m, n, k, alpha, a_ptrs.data(), get_ld(mat_a), b_ptrs.data(), get_ld(mat_b), beta, c_ptrs.data(), get_ld(mat_c),
                          batch_count);
        }
      }
    }
  }

  /**
   * @brief Interface to MKL's/Magma's `gemm_vbatch` routine.
   *
   * @details It simply calls nda::blas::gemm_batch with `is_vbatch` set to true.
   *
   * @tparam A nda::Matrix type.
   * @tparam B nda::Matrix type.
   * @tparam C nda::MemoryMatrix type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param va `std::vector` of input matrices.
   * @param vb `std::vector` of input matrices.
   * @param beta Input scalar \f$ \beta \f$.
   * @param vc `std::vector` of input/output matrices.
   */
  template <Matrix A, Matrix B, MemoryMatrix C>
  void gemm_vbatch(get_value_t<A> alpha, std::vector<A> const &va, std::vector<B> const &vb, get_value_t<A> beta, std::vector<C> &vc) {
    gemm_batch<true>(alpha, va, vb, beta, vc);
  }

  /**
   * @brief Interface to MKL's/CUDA's `gemm_batch_strided` routine.
   *
   * @details This function is similar to nda::blas::gemm_batch except that it takes 3-dimensional arrays as arguments
   * instead of vectors of matrices. The first dimension of the arrays indexes the matrices to be multiplied.
   *
   * @tparam A nda::ArrayOfRank<3> type.
   * @tparam B nda::ArrayOfRank<3> type.
   * @tparam C nda::ArrayOfRank<3> type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param a 3-dimensional input array.
   * @param b 3-dimensional input array.
   * @param beta Input scalar \f$ \beta \f$.
   * @param c 3-dimensional input/output array.
   */
  template <ArrayOfRank<3> A, ArrayOfRank<3> B, MemoryArrayOfRank<3> C>
    requires((MemoryArrayOfRank<A, 3> or is_conj_array_expr<A>) and (MemoryArrayOfRank<B, 3> or is_conj_array_expr<B>)
             and have_same_value_type_v<A, B, C> and mem::have_compatible_addr_space<A, B, C> and is_blas_lapack_v<get_value_t<A>>)
  void gemm_batch_strided(get_value_t<A> alpha, A const &a, B const &b, get_value_t<A> beta, C &&c) {
    // check sizes of input arrays (number of matrices) and return if they are empty
    EXPECTS(a.shape()[0] == b.shape()[0] and a.shape()[0] == c.shape()[0]);
    if (a.size() == 0) return;
    auto const batch_count = a.shape()[0];

    // if C is in C-layout, compute the transpose of the product in Fortran order
    if constexpr (has_C_layout<C>) {
      gemm_batch_strided(alpha, transposed_view<1, 2>(b), transposed_view<1, 2>(a), beta, transposed_view<1, 2>(std::forward<C>(c)));
      return;
    } else {
      // get underlying array in case it is given as a conjugate expression
      auto arr_a = get_array(a);
      auto arr_b = get_array(b);

      // get views of the first matrix in the batch
      auto a0 = arr_a(0, nda::range::all, nda::range::all);
      auto b0 = arr_b(0, nda::range::all, nda::range::all);
      auto c0 = c(0, nda::range::all, nda::range::all);

      // check the dimensions of the input/output arrays/views
      auto const [m, k] = a0.shape();
      auto const [l, n] = b0.shape();
      EXPECTS(k == l);
      EXPECTS(m == c0.extent(0));
      EXPECTS(n == c0.extent(1));

      // arrays/views must be BLAS compatible
      EXPECTS(arr_a.indexmap().min_stride() == 1);
      EXPECTS(arr_b.indexmap().min_stride() == 1);
      EXPECTS(c.indexmap().min_stride() == 1);

      // check for conjugate lazy expressions and C-layouts
      char op_a = get_op<is_conj_array_expr<A>, has_C_layout<A>>;
      char op_b = get_op<is_conj_array_expr<B>, has_C_layout<B>>;

      // perform the actual library call
      if constexpr (mem::have_device_compatible_addr_space<A, B, C>) {
#if defined(NDA_HAVE_DEVICE)
        device::gemm_batch_strided(op_a, op_b, m, n, k, alpha, arr_a.data(), get_ld(a0), arr_a.strides()[0], arr_b.data(), get_ld(b0),
                                   arr_b.strides()[0], beta, c.data(), get_ld(c0), c.strides()[0], batch_count);
#else
        compile_error_no_gpu();
#endif
      } else {
        f77::gemm_batch_strided(op_a, op_b, m, n, k, alpha, arr_a.data(), get_ld(a0), arr_a.strides()[0], arr_b.data(), get_ld(b0),
                                arr_b.strides()[0], beta, c.data(), get_ld(c0), c.strides()[0], batch_count);
      }
    }
  }

  /** @} */

} // namespace nda::blas
