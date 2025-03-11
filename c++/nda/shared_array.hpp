// Copyright (c) 2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2024 Simons Foundation
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
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

/**
 * @file shared_array.hpp
 * @brief Provides the class and utilities for arrays in MPI shared memory.
 *
 * This header defines a shared_array alias that specializes basic_array
 * for arrays allocated in MPI shared memory using the nda::mem::mpi_shm_allocator.
 * It also provides helper functions such as get_win(), fence(), and for_each_chunked()
 * to facilitate MPI synchronization and chunked iteration.
 */

#pragma once

#include "./basic_array.hpp"

namespace nda {

  /**
   * @brief Alias for arrays allocated in MPI shared memory.
   *
   * This alias creates a basic_array with:
   *  - ValueType: The type of the data stored in the array.
   *  - Rank: The number of dimensions of the array.
   *  - Layout: The layout policy (default is C_layout).
   *  - Algebra: Set to 'A' for shared arrays.
   *  - ContainerPolicy: Uses heap_basic with mpi_shm_allocator to allocate memory on an MPI shared memory island.
   *
   * @tparam ValueType The type of the elements stored in the array.
   * @tparam Rank The number of dimensions.
   * @tparam Layout The memory layout policy.
   * @tparam ContainerPolicy The container policy for memory allocation.
   */

  template <typename ValueType, int Rank, typename Layout = C_layout, typename ContainerPolicy = heap_basic<mem::mpi_shm_allocator>>
  using shared_array = basic_array<ValueType, Rank, Layout, 'A', ContainerPolicy>;

  /**
   * @brief Extracts the MPI shared memory window from a handle, if available.
   *
   * @tparam H A handle type satisfying mem::Handle.
   * @param h The handle from which to extract the MPI shared window.
   * @return Pointer to an mpi::shared_window<char> if available; nullptr otherwise.
   */
  template <typename H>
    requires mem::Handle<H>
  mpi::shared_window<char> *get_win(H const &h) {
    if constexpr (requires { h.template userdata<mpi::shared_window<char> *>(); }) {
      if (auto win = h.template userdata<mpi::shared_window<char> *>(); win) { return win; }
    }
    return nullptr;
  }

  /**
   * @brief Synchronizes a shared array using the underlying MPI shared window.
   *
   * @tparam ValueType The type of the elements in the array.
   * @tparam Rank The number of dimensions.
   * @tparam LayoutPolicy The memory layout policy.
   * @tparam Algebra The algebra identifier (should be 'A' for shared_array).
   * @tparam ContainerPolicy The container policy used for memory allocation.
   * @param array A const reference to the basic_array to be synchronized.
   */
  template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename ContainerPolicy>
  void fence(basic_array<ValueType, Rank, LayoutPolicy, Algebra, ContainerPolicy> const &array) {
    auto const &sto               = array.storage();
    mpi::shared_window<char> *win = get_win(sto);
    if (win) {
      win->fence();
    } else {
      ASSERT_WITH_MESSAGE(
         requires { sto.template userdata<mpi::shared_window<char> *>(); }, "fence: storage type does not support MPI shared window");
    }
  }

  /**
   * @brief Applies a functor to each chunk of a shared array.
   *
   * This function divides the array (via its index map) into a number of chunks and
   * applies the provided functor to each element in the specified chunk. This is useful
   * for distributed processing over MPI shared memory.
   *
   * @tparam Functor The type of the function or callable object.
   * @tparam ValueType The type of the array elements.
   * @tparam Rank The number of dimensions.
   * @tparam LayoutPolicy The memory layout policy.
   * @param f The functor to apply to each array element.
   * @param array The shared_array on which to operate.
   * @param n_chunks The total number of chunks to divide the array into.
   * @param rank The rank (chunk index) to process.
   */
  template <typename Functor, typename ValueType, int Rank, typename LayoutPolicy>
  void for_each_chunked(Functor &&f, shared_array<ValueType, Rank, LayoutPolicy> &array, long n_chunks, long rank) {
    auto &lay  = array.indexmap();
    auto slice = itertools::chunk_range(0, lay.size(), n_chunks, rank);
    for (int i = slice.first; i < slice.second; ++i) { f(array(nda::_linear_index_t{i})); }
  }

} // namespace nda
