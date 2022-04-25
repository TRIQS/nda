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

#include "./../map.hpp"
#include "./../exceptions.hpp"

// Models ArrayInitializer concept
template <nda::Array A>
struct mpi::lazy<mpi::tag::reduce, A> {

  using value_type      = typename std::decay_t<A>::value_type;
  using const_view_type = decltype(std::declval<const A>()());

  const_view_type rhs; // the rhs array
  mpi::communicator c; // mpi comm
  const int root;      //
  const bool all;
  const MPI_Op op;

  /// compute the shape of the target array
  [[nodiscard]] auto shape() const { return rhs.shape(); }

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

    if constexpr (not mpi::has_mpi_type<value_type>) {
      target = nda::map([this](auto const &x) { return mpi::reduce(x, this->c, this->root, this->all, this->op); })(rhs);
    } else {

      // some checks.
      bool in_place = (target.data() == rhs.data());
      auto sha      = shape();
      if (in_place) {
        if (rhs.size() != target.size()) NDA_RUNTIME_ERROR << "mpi reduce of array : same pointer to data start, but different number of elements !";
      } else { // check no overlap
        if ((c.rank() == root) || all) resize_or_check_if_view(target, sha);
        if (std::abs(target.data() - rhs.data()) < rhs.size()) NDA_RUNTIME_ERROR << "mpi reduce of array : overlapping arrays !";
      }

      void *v_p       = (void *)target.data();
      void *rhs_p     = (void *)rhs.data();
      auto rhs_n_elem = rhs.size();
      auto D          = mpi::mpi_type<value_type>::get();

      if (!all) {
        if (in_place)
          MPI_Reduce((c.rank() == root ? MPI_IN_PLACE : rhs_p), rhs_p, rhs_n_elem, D, op, root, c.get());
        else
          MPI_Reduce(rhs_p, v_p, rhs_n_elem, D, op, root, c.get());
      } else {
        if (in_place)
          MPI_Allreduce(MPI_IN_PLACE, rhs_p, rhs_n_elem, D, op, c.get());
        else
          MPI_Allreduce(rhs_p, v_p, rhs_n_elem, D, op, c.get());
      }
    }
  }
};

namespace nda {

  /**
   * Reduction of the array
   *
   * \tparam A basic_array or basic_array_view, with contiguous data only
   * \param a
   * \param c The MPI communicator
   * \param root Root node of the reduction
   * \param all all_reduce iif true
   * \param op The MPI reduction operation to apply to the elements 
   *
   * NB : A::value_type must have an MPI reduction (basic type or custom type, cf mpi library)
   *
   */
  template <typename A>
  ArrayInitializer<std::remove_reference_t<A>> auto mpi_reduce(A &&a, mpi::communicator c = {}, int root = 0, bool all = false,
                                                               MPI_Op op = MPI_SUM) requires(is_regular_or_view_v<std::decay_t<A>>) {

    if (not a.is_contiguous()) NDA_RUNTIME_ERROR << "mpi operations require contiguous rhs.data() to be contiguous";

    return mpi::lazy<mpi::tag::reduce, A>{std::forward<A>(a), c, root, all, op};
  }

} // namespace nda
