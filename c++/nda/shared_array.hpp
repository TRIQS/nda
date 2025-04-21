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
 * for arrays allocated in MPI shared memory using the nda::mem::mallocator.
 * It also provides helper functions such as get_win(), fence(), and for_each_chunked()
 * to facilitate MPI synchronization and chunked iteration.
 */

#pragma once

#include "./basic_array.hpp"

namespace nda {

  /// Concept for a valid shared array.
  /// storage array has userdata in addition
  template <typename Array>
  concept SharedArray = requires(Array a) {
    { a.storage().userdata() } -> std::convertible_to<mpi::shared_window<char> *>;
  } && Array::storage_t::address_space == mem::MPISharedMemory;

  /**
   * @addtogroup shared_av_types
   * @{
   */

  /**
   * @brief Alias for arrays allocated in MPI shared memory.
   *
   * This alias creates a basic_array with:
   *  - ValueType: The type of the data stored in the array.
   *  - Rank: The number of dimensions of the array.
   *  - Layout: The layout policy (default is C_layout).
   *  - Algebra: Set to 'A' for shared arrays.
   *
   * @tparam ValueType The type of the elements stored in the array.
   * @tparam Rank The number of dimensions.
   * @tparam Layout The memory layout policy.
   */

  template <typename ValueType, int Rank, typename Layout = C_layout>
  using shared_array = basic_array<ValueType, Rank, Layout, 'A', heap<mem::MPISharedMemory>>;

  /**
   * @brief Alias template of an nda::shared_array_view with an 'A' algebra, nda::default_accessor and nda::borrowed
   * owning policy.
   *
   * @tparam ValueType Value type of the view.
   * @tparam Rank Rank of the view.
   * @tparam Layout Layout policy of the view.
   */
  template <typename ValueType, int Rank, typename Layout = C_stride_layout>
  using shared_array_view = basic_array_view<ValueType, Rank, Layout, 'A', default_accessor, borrowed<mem::MPISharedMemory>>;

  /**
   * @brief Same as shared_array_view except for const value types.
   *
   * @tparam ValueType Value type of the view.
   * @tparam Rank Rank of the view.
   * @tparam Layout Layout policy of the view.
   */
  template <typename ValueType, int Rank, typename Layout = C_stride_layout>
  using shared_array_const_view = basic_array_view<ValueType const, Rank, Layout, 'A', default_accessor, borrowed<mem::MPISharedMemory>>;

  /**
   * @brief Alias template for a contiguous shared array view.
   *
   * This alias is a specialization of basic_array_view that represents a contiguous shared array view in MPI shared memory.
   * It requires the layout to have a contiguous memory mapping.
   *
   * @tparam ValueType The type of the elements in the view.
   * @tparam Rank The number of dimensions in the view.
   * @tparam Layout The memory layout policy (default is C_layout) that must have contiguous memory layout properties.
   */
  template <typename ValueType, int Rank, typename Layout = C_stride_layout>
    requires(has_contiguous(Layout::template mapping<Rank>::layout_prop))
  using shared_array_contiguous_view = basic_array_view<ValueType, Rank, Layout, 'A', default_accessor, borrowed<mem::MPISharedMemory>>;

  /**
   * @brief Alias template for a contiguous shared array const view.
   *
   * This alias is a specialization of basic_array_view that represents a contiguous shared array view with constant elements in MPI shared memory.
   * It requires the layout to guarantee contiguous memory storage.
   *
   * @tparam ValueType The type of the elements in the view.
   * @tparam Layout The memory layout policy (default is C_stride_layout) that must have contiguous memory layout properties.
   */
  template <typename ValueType, int Rank, typename Layout = C_stride_layout>
    requires(has_contiguous(Layout::template mapping<Rank>::layout_prop))
  using shared_array_contiguous_const_view = basic_array_view<ValueType const, Rank, Layout, 'A', default_accessor, borrowed<mem::MPISharedMemory>>;

  /**
   * @brief Alias for matrices allocated in MPI shared memory.
   *
   * This alias creates a basic_array with:
   *  - ValueType: The type of the data stored in the matrix.
   *  - Rank: The number of dimensions.
   *  - Layout: The memory layout policy (default is C_layout).
   *  - Algebra: Set to 'M' for matrix algebra.
   *  - ContainerPolicy: Uses heap_basic with mallocator to allocate memory on an MPI shared memory island.
   *
   * @tparam ValueType The type of the elements stored in the matrix.
   * @tparam Layout The memory layout policy.
   */
  template <typename ValueType, typename Layout = C_layout>
  using shared_matrix = basic_array<ValueType, 2 , Layout, 'M', heap<mem::MPISharedMemory>>;

  /**
   * @brief Alias template for a shared matrix view.
   *
   * This alias represents a non-owning view of a matrix allocated in MPI shared memory.
   * It uses the default accessor and a borrowed owning policy with MPI shared memory settings.
   *
   * @tparam ValueType The type of the elements in the view.
   * @tparam Layout The memory layout policy (default is C_stride_layout).
   */
  template <typename ValueType, typename Layout = C_stride_layout>
  using shared_matrix_view = basic_array_view<ValueType, 2, Layout, 'A', default_accessor, borrowed<mem::MPISharedMemory>>;

  /**
   * @brief Alias template for a const shared matrix view.
   *
   * This alias represents a non-owning view of a matrix with constant elements allocated in MPI shared memory.
   * It employs matrix algebra ('M') along with the default accessor and a borrowed owning policy.
   *
   * @tparam ValueType The type of the elements in the view.
   * @tparam Layout The memory layout policy (default is C_stride_layout).
   */
  template <typename ValueType, typename Layout = C_stride_layout>
  using shared_matrix_const_view = basic_array_view<ValueType const, 2, Layout, 'M', default_accessor, borrowed<mem::MPISharedMemory>>;

  /**
   * @brief Alias for vectors allocated in MPI shared memory.
   *
   * This alias creates a basic_array specifically for vector algebra, with:
   *  - ValueType: The type of the data stored in the vector.
   *  - Rank: The number of dimensions (usually 1 for vectors).
   *  - Layout: The memory layout policy (default is C_layout).
   *  - Algebra: Set to 'V' for vector algebra.
   *  - ContainerPolicy: Uses heap_basic with mallocator to allocate memory on an MPI shared memory island.
   *
   * @tparam ValueType The type of the elements stored in the vector.
   */
  template <typename ValueType>
  using shared_vector = basic_array<ValueType, 1, C_layout, 'V', heap<mem::MPISharedMemory>>;

  /**
   * @brief Alias template for a shared vector view.
   *
   * This alias represents a non-owning view of a vector allocated in MPI shared memory.
   * It uses the default accessor and a borrowed owning policy with MPI shared memory.
   *
   * @tparam ValueType The type of the elements in the view.
   * @tparam Rank The number of dimensions of the view.
   * @tparam Layout The memory layout policy (default is C_stride_layout).
   */
  template <typename ValueType, typename Layout = C_stride_layout>
  using shared_vector_view = basic_array_view<ValueType, 1, Layout, 'V', default_accessor, borrowed<mem::MPISharedMemory>>;

  /**
   * @brief Alias template for a const shared vector view.
   *
   * This alias represents a non-owning view of a vector with constant elements allocated in MPI shared memory.
   * It uses the default accessor and a borrowed owning policy with MPI shared memory.
   *
   * @tparam ValueType The type of the elements in the view.
   * @tparam Rank The number of dimensions of the view.
   * @tparam Layout The memory layout policy (default is C_stride_layout).
   */
  template <typename ValueType, typename Layout = C_stride_layout>
  using shared_vector_const_view = basic_array_view<ValueType const, 1, Layout, 'V', default_accessor, borrowed<mem::MPISharedMemory>>;

  /** @} */

  /**
   * @addtogroup shared_av_utils
   * @{
   */

  /**
   * @brief Get the type of the nda::shared_array that would be obtained by constructing an array from a given type.
   * @tparam T Type to construct an array from.
   */
  template <typename T, typename T2 = std::remove_reference_t<T> /* Keep this: Fix for gcc11 bug */>
  using get_regular_t = decltype(basic_array{std::declval<T>()});

  /**
    * @brief Get the type of the nda::basic_array that would be obtained by constructing an array on host memory from a
    * given type.
    *
    * @tparam T Type to construct an array from.
    */
  template <typename T, typename RT = get_regular_t<T>>
  using get_regular_shm_t =
     std::conditional_t<mem::on_mpi_shared_memory<RT>, RT,
                        shared_array<get_value_t<RT>, get_rank<RT>, get_contiguous_layout_policy<get_rank<RT>, get_layout_info<RT>.stride_order>>>;

  /** @} */

  /**
   * @brief Extracts the MPI shared memory window from a shared array with correct policy.
   *
   * @tparam ValueType The type of the elements in the array.
   * @tparam Rank The number of dimensions.
   * @tparam LayoutPolicy The memory layout policy.
   * @tparam Algebra The algebra identifier (should be 'A' for shared_array).
   * @tparam ContainerPolicy The container policy used for memory allocation.
   * @param array A const reference to the basic_array.
   * @return Pointer to an mpi::shared_window<char> if available; nullptr otherwise.
   */
  template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename ContainerPolicy>
  //requires SharedArray<basic_array<ValueType, Rank, LayoutPolicy, Algebra, ContainerPolicy>>
  mpi::shared_window<char> *get_window(basic_array<ValueType, Rank, LayoutPolicy, Algebra, ContainerPolicy> const &array) {
    auto const &sto = array.storage();
    if constexpr (requires { sto.userdata(); }) { return sto.userdata(); }
    return nullptr;
  }

  /**
   * @brief Extracts the MPI shared memory window from a shared array with correct policy.
   *
   * @tparam ValueType The type of the elements in the array.
   * @tparam Rank The number of dimensions.
   * @tparam LayoutPolicy The memory layout policy.
   * @tparam Algebra The algebra identifier (should be 'A' for shared_array).
   * @tparam AccessorPolicy Policy determining how the data pointer is accessed.
   * @tparam OwningPolicy Policy determining the ownership of the data.
   * @param array A const reference to the basic_array_view.
   * @return Pointer to an mpi::shared_window<char> if available; nullptr otherwise.
   */
  template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  //requires SharedArray<basic_array_view<ValueType, Rank, LayoutPolicy, Algebra, OwningPolicy>>
  mpi::shared_window<char> *get_window(basic_array_view<ValueType, Rank, LayoutPolicy, Algebra, AccessorPolicy, OwningPolicy> const &array_view) {
    auto const &sto = array_view.storage();
    if constexpr (requires { sto.userdata(); }) { return sto.userdata(); }
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
   * @param array A const reference to the basic_array.
   */
  template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename ContainerPolicy>
  void fence(basic_array<ValueType, Rank, LayoutPolicy, Algebra, ContainerPolicy> const &array) {
    mpi::shared_window<char> *win = get_window(array);
    ASSERT(win != nullptr);
    win->fence();
  }

  /**
   * @brief Synchronizes a shared array using the underlying MPI shared window.
   *
   * @tparam ValueType The type of the elements in the array.
   * @tparam Rank The number of dimensions.
   * @tparam LayoutPolicy The memory layout policy.
   * @tparam Algebra The algebra identifier (should be 'A' for shared_array).
   * @tparam Algebra The algebra identifier (should be 'A' for shared_array).
   * @tparam AccessorPolicy Policy determining how the data pointer is accessed.
   * @tparam OwningPolicy Policy determining the ownership of the data.
   * @param array A const reference to the basic_array.
   */
  template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  void fence(basic_array_view<ValueType, Rank, LayoutPolicy, Algebra, AccessorPolicy, OwningPolicy> const &array_view) {
    mpi::shared_window<char> *win = get_window(array_view);
    ASSERT(win != nullptr);
    win->fence();
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
   * @tparam AccessorPolicy Policy determining how the data pointer is accessed.
   * @tparam OwningPolicy Policy determining the ownership of the data.
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
