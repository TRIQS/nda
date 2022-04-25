// Copyright (c) 2020-2021 Simons Foundation
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

#include "../concepts.hpp"
#include "../exceptions.hpp"

// Models ArrayInitializer concept
template <nda::Array A>
struct mpi::lazy<mpi::tag::scatter, A> {

  using value_type      = typename std::decay_t<A>::value_type;
  using const_view_type = decltype(std::declval<const A>()());

  const_view_type rhs; // the rhs array
  mpi::communicator c; // mpi comm
  const int root;      //
  const bool all;

  /// compute the shape of the target array. WARNING : MAKES A MPI CALL.
  [[nodiscard]] auto shape() const {
    auto dims      = rhs.shape();
    long slow_size = dims[0];
    mpi::broadcast(slow_size, c, root);
    dims[0] = mpi::chunk_length(slow_size, c.size(), c.rank());
    return dims;
  }

  /// Execute the mpi operation and write result to target
  template <nda::Array T>
  void invoke(T &&target) const {
    if (not target.is_contiguous()) NDA_RUNTIME_ERROR << "mpi operations require contiguous target.data() to be contiguous";

    static_assert(std::decay_t<A>::layout_t::stride_order_encoded == std::decay_t<T>::layout_t::stride_order_encoded,
                  "Array types for rhs and target have incompatible stride order");

    if (not mpi::has_env) {
      target = rhs;
      return;
    }

    auto sha = shape(); // WARNING : Keep this out of any if condition (shape USES MPI) !
    resize_or_check_if_view(target, sha);

    auto slow_size   = rhs.extent(0);
    auto slow_stride = rhs.indexmap().strides()[0];
    auto sendcounts  = std::vector<int>(c.size());
    auto displs      = std::vector<int>(c.size() + 1, 0);
    int recvcount    = mpi::chunk_length(slow_size, c.size(), c.rank()) * slow_stride;
    auto D           = mpi::mpi_type<value_type>::get();

    for (int r = 0; r < c.size(); ++r) {
      sendcounts[r] = mpi::chunk_length(slow_size, c.size(), r) * slow_stride;
      displs[r + 1] = sendcounts[r] + displs[r];
    }

    MPI_Scatterv((void *)rhs.data(), &sendcounts[0], &displs[0], D, (void *)target.data(), recvcount, D, root, c.get());
  }
};

namespace nda {

  /**
   * Scatter the array over mpi threads
   *
   * \tparam A basic_array or basic_array_view, with contiguous data only
   * \param a
   * \param c The MPI communicator
   * \param root Root node of the reduction
   * \param all all_reduce iif true
   *
   * NB : A::value_type must have an MPI reduction (basic type or custom type, cf mpi library)
   */
  template <typename A>
  ArrayInitializer<std::remove_reference_t<A>> auto mpi_scatter(A &&a, mpi::communicator c = {}, int root = 0,
                                                                bool all = false) requires(is_regular_or_view_v<std::decay_t<A>>) {

    if (not a.is_contiguous()) NDA_RUNTIME_ERROR << "mpi operations require contiguous rhs.data() to be contiguous";

    return mpi::lazy<mpi::tag::scatter, A>{std::forward<A>(a), c, root, all};
  }

} // namespace nda
