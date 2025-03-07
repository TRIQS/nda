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

template <typename ValueType, int Rank, typename Layout = C_layout, typename ContainerPolicy = heap_basic<mem::mpi_shm_allocator>>
using shared_array = basic_array<ValueType, Rank, Layout, 'A', ContainerPolicy>;

/// Helper function at compile time: attempts to get MPI window (if it exists)
template <typename H>
concept HasUserdata = mem::Handle<H> && requires(H h) {
  { h.template userdata<mpi::shared_window<char>*>() } -> std::convertible_to<mpi::shared_window<char>*>;
};

/// Helper function: Extracts the MPI shared window from a handle, if available.
template <typename H>
  requires mem::Handle<H>
mpi::shared_window<char>* get_win(H const& h) {
  if constexpr (HasUserdata<H>) {
    if (auto win = h.template userdata<mpi::shared_window<char>*>(); win) {
      return win;
    }
  }

  if constexpr (requires { h.parent(); }) {
    if (auto parent = h.parent(); parent) {
      if constexpr (HasUserdata<decltype(*parent)>) {
        return parent->template userdata<mpi::shared_window<char>*>();
      }
    }
  }
  return nullptr;
}

template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename ContainerPolicy>
void fence(basic_array<ValueType, Rank, LayoutPolicy, Algebra, ContainerPolicy> const &array) {
  auto sto = array.storage();
  mpi::shared_window<char>* win = get_win(sto);
  if (win) {
    win->fence();
  } else {
    static_assert(HasUserdata<decltype(sto)>, "fence: storage type does not support MPI shared window");
  }
}

template <typename Functor, typename ValueType, int Rank, typename LayoutPolicy>
void for_each_chunked(Functor &&f, shared_array<ValueType, Rank, LayoutPolicy> &array, long n_chunks, long rank) {
  auto &lay = array.indexmap();
  auto slice = itertools::chunk_range(0, lay.size(), n_chunks, rank);
  for (int i = slice.first; i < slice.second; ++i) {
    f(array(nda::_linear_index_t{i}));
  }
}

} // namespace nda
