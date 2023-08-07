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
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once
#include <mpi/mpi.hpp>
#include "./mpi/reduce.hpp"
#include "./mpi/scatter.hpp"
#include "./mpi/gather.hpp"

namespace nda {

  /**
   *  Broadcast the array
   *
   * \tparam A basic_array or basic_array_view, with contiguous data only
   * \param a
   * \param c The MPI communicator
   * \param root Root node of the reduction
   * \param all all_reduce iif true
   * \param op The MPI reduction operation to apply to the elements 
   */
  template <typename A>
  void mpi_broadcast(A &a, mpi::communicator c = {}, int root = 0) //
    requires(is_regular_or_view_v<A>)
  {
    static_assert(has_contiguous_layout<A>, "Non contigous view in mpi_broadcast are not implemented");
    auto sh = a.shape();
    //FIXME : mpi::std::array
    MPI_Bcast(&sh[0], sh.size(), mpi::mpi_type<typename decltype(sh)::value_type>::get(), root, c.get());
    if (c.rank() != root) { resize_or_check_if_view(a, sh); }
    MPI_Bcast(a.data(), a.size(), mpi::mpi_type<typename A::value_type>::get(), root, c.get());
  }

} // namespace nda
