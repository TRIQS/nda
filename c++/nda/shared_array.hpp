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
 * @file
 * @brief Provides the class for arrays in MPI shared memory.
 */

#pragma once

#include "./basic_array.hpp"

namespace nda {

template <typename ValueType, int Rank, typename LayoutPolicy = C_layout, char Algebra = 'A'>
class shared_array : public basic_array<ValueType, Rank, LayoutPolicy, Algebra, nda::mpi_shared_memory<nda::mem::mpi_shm_allocator>> {
public:
  using base_t = basic_array<ValueType, Rank, LayoutPolicy, Algebra, nda::mpi_shared_memory<nda::mem::mpi_shm_allocator>>;
  using layout_t = base_t::layout_t;
  using storage_t = base_t::storage_t;
private:
  mpi::shared_communicator _c{mpi::communicator{}.split_shared()};
public:
  shared_array() : base_t() {};

  shared_array(mpi::shared_communicator c) : base_t(), _c(c) {};

  template <std::integral Int = long>
  explicit shared_array(std::array<Int, Rank> const &shape) : base_t() {}

  template <std::integral Int = long>
  explicit shared_array(std::array<Int, Rank> const &shape, mpi::shared_communicator c) : base_t(layout_t{shape}, storage_t{layout_t{shape}.size(), c}), _c(c) {}

  explicit shared_array(const shared_array&) = default;

  shared_array& operator=(const shared_array&) = default;

  shared_array(shared_array&& other) = default;

  shared_array& operator=(shared_array&& other) = default;

  mpi::shared_communicator comm() const { return _c; }

  mpi::shared_window<char> &win() const { return *static_cast<mpi::shared_window<char>*>(this->storage().userdata()); }
};

template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra>
void fence(shared_array<ValueType, Rank, LayoutPolicy, Algebra> const &array) {
  array.win().fence();
}

template <typename Functor, typename ValueType, int Rank, typename LayoutPolicy, char Algebra>
void for_each_chunked(Functor &&f, shared_array<ValueType, Rank, LayoutPolicy, Algebra> &array, long n_chunks, long rank) {
  auto &lay = array.indexmap();
  auto slice = itertools::chunk_range(0, lay.size(), n_chunks, rank);
  for (int i = slice.first; i < slice.second; ++i) {
    f(array(nda::_linear_index_t{i}));
  }
}

} // namespace nda
