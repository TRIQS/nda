// Copyright (c) 2022--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to batched versions of the BLAS/cuBLAS `gemm` routine.
 */

#pragma once

#include "./interface/cxx_interface.hpp"
#include "./tools.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../device.hpp"
#include "../layout_transforms.hpp"
#include "../macros.hpp"
#include "../mem/address_space.hpp"
#include "../traits.hpp"

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
    template <bool is_vbatch, nda::mem::AddressSpace vec_addr_spc>
    auto get_ptr_vector(auto &&v) {
      EXPECTS(std::ranges::all_of(v, [&v](auto &A) { return is_vbatch or A.shape() == v[0].shape(); }));
      EXPECTS(std::ranges::all_of(v, [](auto &A) { return get_array(A).indexmap().min_stride() == 1; }));
      using ptr_t = std::remove_reference_t<decltype(get_first_element(v[0]))> *;
      auto v_ptrs = nda::vector<ptr_t, heap<vec_addr_spc>>(v.size());
      std::transform(v.begin(), v.end(), v_ptrs.begin(), [](auto &z) { return get_array(z).data(); });
      return v_ptrs;
    }

  } // namespace detail

  /**
   * @brief Interface to batched versions of the BLAS/cuBLAS `gemm` routine.
   *
   * @details This function performs the matrix-matrix operations
   * \f[
   *   \mathbf{C}_i \leftarrow \alpha \mathbf{A}_i \mathbf{B}_i + \beta \mathbf{C}_i \;,
   * \f] 
   * for batches of matrices indexed by \f$ i \in \{ 0, \ldots, N_b - 1 \} \f$. Here, \f$ N_b \f$ is the batch size and 
   * \f$ \alpha \f$ and \f$ \beta \f$ are scalars. See also nda::blas::gemm.
   *
   * A batch of matrices is just a `std::vector` of nda::blas_lapack::BlasArray or nda::blas_lapack::BlasArrayOrConj 
   * objects. If `is_vbatch` is true, the matrices are allowed to have different sizes. Otherwise, they are required to 
   * have the same size.
   * 
   * Depending on the types of input matrices, the template parameter `is_vbatch` and the availability of MAGMA and MKL 
   * libraries, the function does the following:
   * - If the input matrices satisfy nda::mem::have_device_compatible_addr_space and 
   *   - `is_vbatch` is false, it calls cuBLAS's `cublasXgemmBatched`. 
   *   - `is_vbatch` is true and 
   *     - the matrices are real, it calls cuBLAS's `cublasGemmGroupedBatchedEx`
   *     - the matrices are complex, it (tries) to call `magmablas_Xgemm_vbatched`. If **nda** has not been configured
   *     with MAGMA support, an exception is thrown.
   * - If the input matrices do not satisfy nda::mem::have_device_compatible_addr_space and
   *   - **nda** is linked to MKL, it calls MKL's `Xgemm_batch` for both `is_vbatch` true and false.
   *   - **nda** is not linked to MKL, it simply loops over all matrices in the batch and calls nda::blas::gemm.
   * 
   * @note \f$ \mathbf{A}_i \f$ and \f$ \mathbf{B}_i \f$ are allowed to be lazy conjugate expressions (see 
   * nda::blas_lapack::is_conj_array_expr). In this case, they are required to have the opposite memory layout of \f$ 
   * \mathbf{C}_i \f$ (see nda::C_layout vs nda::F_layout).
   *
   * @tparam is_vbatch Allow variable sized matrices.
   * @tparam A nda::blas_lapack::BlasArrayOrConj<2> type.
   * @tparam B nda::blas_lapack::BlasArrayOrConjFor<A, 2> type.
   * @tparam C nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param va `std::vector` of size \f$ N_b \f$ containing the input matrices \f$ \mathbf{A}_i \f$.
   * @param vb `std::vector` of size \f$ N_b \f$ containing the input matrices \f$ \mathbf{B}_i \f$.
   * @param beta Input scalar \f$ \beta \f$.
   * @param vc `std::vector` of size \f$ N_b \f$ containing the input/output matrices \f$ \mathbf{C}_i \f$.
   */
  template <bool is_vbatch = false, BlasArrayOrConj<2> A, BlasArrayOrConjFor<A, 2> B, BlasArrayFor<A, 2> C>
  void gemm_batch(get_value_t<A> alpha, std::vector<A> const &va, std::vector<B> const &vb, get_value_t<A> beta, std::vector<C> &vc) {
    auto const n_b = va.size();

    // check sizes of input vectors and return if they are empty
    EXPECTS(n_b == vb.size() and n_b == vc.size());
    if (va.empty()) return;

    // if C is in C-layout, compute the transpose of the product in Fortran order
    if constexpr (has_C_layout<C>) {
      auto vcT = detail::get_transpose_vector(vc);
      return gemm_batch<is_vbatch>(alpha, detail::get_transpose_vector(vb), detail::get_transpose_vector(va), beta, vcT);
    } else {
      // for operations on the device, use unified memory for vector of ints or ptrs
      auto constexpr vec_addr_spc = []() { return mem::on_host<C> ? mem::Host : mem::Unified; }();

      // convert the vector of matrices to the corresponding vector of pointers
      auto a_ptrs = detail::get_ptr_vector<is_vbatch, vec_addr_spc>(va);
      auto b_ptrs = detail::get_ptr_vector<is_vbatch, vec_addr_spc>(vb);
      auto c_ptrs = detail::get_ptr_vector<is_vbatch, vec_addr_spc>(vc);

      // either call gemm_vbatch or gemm_batch
      if constexpr (is_vbatch) {
        // create vectors to store shapes and leading dimensions of size 'batch_count + 1' as required by Magma
        nda::vector<int, heap<vec_addr_spc>> vm(n_b + 1), vk(n_b + 1), vn(n_b + 1), vlda(n_b + 1), vldb(n_b + 1), vldc(n_b + 1);

        for (auto i : range(n_b)) {
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
          device::gemm_vbatch(get_op<A>, get_op<B>, vm.data(), vn.data(), vk.data(), alpha, a_ptrs.data(), vlda.data(), b_ptrs.data(), vldb.data(),
                              beta, c_ptrs.data(), vldc.data(), n_b);
        } else {
          f77::gemm_vbatch(get_op<A>, get_op<B>, vm.data(), vn.data(), vk.data(), alpha, a_ptrs.data(), vlda.data(), b_ptrs.data(), vldb.data(), beta,
                           c_ptrs.data(), vldc.data(), n_b);
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
          device::gemm_batch(get_op<A>, get_op<B>, m, n, k, alpha, a_ptrs.data(), get_ld(mat_a), b_ptrs.data(), get_ld(mat_b), beta, c_ptrs.data(),
                             get_ld(mat_c), n_b);
        } else {
          f77::gemm_batch(get_op<A>, get_op<B>, m, n, k, alpha, a_ptrs.data(), get_ld(mat_a), b_ptrs.data(), get_ld(mat_b), beta, c_ptrs.data(),
                          get_ld(mat_c), n_b);
        }
      }
    }
  }

  /**
   * @brief Interface to batched versions of the BLAS/cuBLAS `gemm` routine for variable sized matrices.
   *
   * @details It simply calls nda::blas::gemm_batch with `is_vbatch` set to true.
   *
   * @tparam A nda::blas_lapack::BlasArrayOrConj<2> type.
   * @tparam B nda::blas_lapack::BlasArrayOrConjFor<A, 2> type.
   * @tparam C nda::blas_lapack::BlasArrayFor<A, 2> type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param va `std::vector` of size \f$ N_b \f$ containing the input matrices \f$ \mathbf{A}_i \f$.
   * @param vb `std::vector` of size \f$ N_b \f$ containing the input matrices \f$ \mathbf{B}_i \f$.
   * @param beta Input scalar \f$ \beta \f$.
   * @param vc `std::vector` of size \f$ N_b \f$ containing the input/output matrices \f$ \mathbf{C}_i \f$.
   */
  template <BlasArrayOrConj<2> A, BlasArrayOrConjFor<A, 2> B, BlasArrayFor<A, 2> C>
  void gemm_vbatch(get_value_t<A> alpha, std::vector<A> const &va, std::vector<B> const &vb, get_value_t<A> beta, std::vector<C> &vc) {
    gemm_batch<true>(alpha, va, vb, beta, vc);
  }

  /**
   * @brief Interface to batched-strided versions of the BLAS/cuBLAS `gemm` routine.
   *
   * @details This function performs the matrix-matrix operations
   * \f[
   *   \mathbf{C}_i \leftarrow \alpha \mathbf{A}_i \mathbf{B}_i + \beta \mathbf{C}_i \;,
   * \f] 
   * for batches of matrices indexed by \f$ i \in \{ 0, \ldots, N_b - 1 \} \f$. Here, \f$ N_b \f$ is the batch size and 
   * \f$ \alpha \f$ and \f$ \beta \f$ are scalars. See also nda::blas::gemm.
   * 
   * A batch of matrices is just a 3-dimensional array in either nda::C_layout or nda::F_layout. For a Fortran (C) 
   * layout array, the last (first) dimension indexes the individual matrices such that `M(:,:,i)` (`M(i,:,:)`) 
   * corresponds to the \f$ i \f$-th matrix \f$ \mathbf{M}_i \f$ in the batch.
   * 
   * Depending on the types of input arrays and the availability of the MKL library, the function does the following:
   * - If the input arrays satisfy nda::mem::have_device_compatible_addr_space, it calls cuBLAS's 
   * `cublasXgemmStridedBatched`.
   * - If the input arrays do not satisfy nda::mem::have_device_compatible_addr_space and
   *   - **nda** is linked to MKL, it calls MKL's `Xgemm_batch_strided`.
   *   - **nda** is not linked to MKL, it simply loops over all matrices in the batch and calls nda::blas::gemm.
   * 
   * @note The 3-dimensional arrays \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ are allowed to be lazy conjugate 
   * expressions (see nda::blas_lapack::is_conj_array_expr). In this case, they are required to have the opposite memory 
   * layout of \f$ \mathbf{C} \f$ (see nda::C_layout vs nda::F_layout).
   *
   * @tparam A nda::blas_lapack::BlasArrayOrConj<3> type.
   * @tparam B nda::blas_lapack::BlasArrayOrConjFor<A, 3> type.
   * @tparam C nda::blas_lapack::BlasArrayFor<A, 3> type.
   * @param alpha Input scalar \f$ \alpha \f$.
   * @param a 3-dimensional input array \f$ \mathbf{A} \f$ containing the matrices \f$ \mathbf{A}_i \f$.
   * @param b 3-dimensional input array \f$ \mathbf{B} \f$ containing the matrices \f$ \mathbf{B}_i \f$.
   * @param beta Input scalar \f$ \beta \f$.
   * @param c 3-dimensional input/output array \f$ \mathbf{C} \f$ containing the matrices \f$ \mathbf{C}_i \f$.
   */
  template <BlasArrayOrConj<3> A, BlasArrayOrConjFor<A, 3> B, BlasArrayFor<A, 3> C>
    requires((has_C_layout<A> or has_F_layout<A>) and (has_C_layout<B> or has_F_layout<B>) and (has_C_layout<C> or has_F_layout<C>))
  void gemm_batch_strided(get_value_t<A> alpha, A const &a, B const &b, get_value_t<A> beta, C &&c) {
    // if C is in C-layout, compute the transpose of the product in Fortran order
    if constexpr (has_C_layout<C>) {
      gemm_batch_strided(alpha, transpose(b), transpose(a), beta, transpose(std::forward<C>(c)));
    } else {
      // get array info: batch count, matrix dims, leading dims, slowest stride
      auto array_info = [](auto &arr) {
        if constexpr (has_C_layout<decltype(arr)>) {
          auto mat = arr(0, nda::ellipsis{});
          return std::array<long, 5>{arr.extent(0), mat.extent(0), mat.extent(1), get_ld(mat), arr.strides()[0]};
        } else {
          auto mat = arr(nda::ellipsis{}, 0);
          return std::array<long, 5>{arr.extent(2), mat.extent(0), mat.extent(1), get_ld(mat), arr.strides()[2]};
        }
      };

      // get underlying array in case it is given as a conjugate expression
      auto arr_a = get_array(a);
      auto arr_b = get_array(b);

      // check the dimensions of the input/output arrays/views
      auto const [nb_a, m_a, k_a, ld_a, s_a] = array_info(arr_a);
      auto const [nb_b, k_b, n_b, ld_b, s_b] = array_info(arr_b);
      auto const [nb_c, m_c, n_c, ld_c, s_c] = array_info(c);
      EXPECTS(k_a == k_b);
      EXPECTS(m_a == m_c);
      EXPECTS(n_b == n_c);
      EXPECTS(nb_a == nb_b and nb_a == nb_c);

      // arrays/views must be BLAS compatible
      EXPECTS(arr_a.indexmap().min_stride() == 1);
      EXPECTS(arr_b.indexmap().min_stride() == 1);
      EXPECTS(c.indexmap().min_stride() == 1);

      // perform the actual library call
      if constexpr (mem::have_device_compatible_addr_space<A, B, C>) {
        device::gemm_batch_strided(get_op<A>, get_op<B>, m_c, n_c, k_a, alpha, arr_a.data(), ld_a, s_a, arr_b.data(), ld_b, s_b, beta, c.data(), ld_c,
                                   s_c, nb_c);
      } else {
        f77::gemm_batch_strided(get_op<A>, get_op<B>, m_c, n_c, k_a, alpha, arr_a.data(), ld_a, s_a, arr_b.data(), ld_b, s_b, beta, c.data(), ld_c,
                                s_c, nb_c);
      }
    }
  }

  /** @} */

} // namespace nda::blas
