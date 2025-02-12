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
  private:
    using Base = basic_array<ValueType, Rank, LayoutPolicy, Algebra, nda::mpi_shared_memory<nda::mem::mpi_shm_allocator>>;
    mpi::shared_communicator _c{mpi::communicator{}.split_shared()};
  public:
    using Base::Base;

    shared_array() : _c(mpi::communicator{}.split_shared()) {};

    shared_array(mpi::shared_communicator c) : _c(c) {};

    shared_array(mpi::shared_communicator c, std::array<long, Rank> const &shape,mem::do_not_initialize_t tag = mem::do_not_initialize)
    : Base(shape, nda::mpi_shared_memory<nda::mem::mpi_shm_allocator>{c}, tag), _c(c) {}

    shared_array(const shared_array&) = delete;

    shared_array& operator=(const shared_array&) = delete;

    shared_array(shared_array&& other) noexcept : Base(std::move(other)), _c(std::move(other._c)) {
      other.reset();
    };

    shared_array& operator=(shared_array&& other) noexcept {
      if (this != &other) {
        Base::operator=(std::move(other));
        _c = std::move(other._c);
        other.reset();
    }
    return *this;
    };

    // TODO: Fix this, check move constructor in communicator
    void reset() {
      Base::operator=(Base{});
      _c = mpi::shared_communicator{}.split_shared();
    }
  };
} // namespace nda
