// Copyright (c) 2020 Simons Foundation
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

#pragma once
#include <mpi/mpi.hpp>

#include "./../map.hpp"

namespace nda::lazy_mpi {

  // Models ArrayInitializer concept
  template <typename ValueType, int Rank, uint64_t StrideOrder>
  struct reduce {

    using view_t = basic_array_view<ValueType const, Rank, basic_layout<0, StrideOrder, layout_prop_e::contiguous>, 'A', default_accessor, borrowed>;
    using value_type = ValueType;

    view_t source;       // view of the array to reduce
    mpi::communicator c; // mpi comm
    const int root;      //
    const bool all;
    const MPI_Op op;

    /// compute the shape of the target array
    [[nodiscard]] auto shape() const { return source.shape(); }

    /// Delayed reduction operation
    void invoke(array_view<ValueType, Rank> target) const {
      // we force the caller to build a view_t. If not possible, e.g. stride orders mismatch, it will not compile

      if constexpr(not mpi::has_mpi_type<value_type>){
	target = nda::map([this](value_type const & x){ return mpi::reduce(x, this->c, this->root, this->all, this->op); })(source);
      } else {

        view_t target_view{target};
        // some checks.
        bool in_place = (target_view.data() == source.data());
        auto sha      = shape();
        if (in_place) {
          if (source.size() != target_view.size())
            NDA_RUNTIME_ERROR << "mpi reduce of array : same pointer to data start, but different number of elements !";
        } else { // check no overlap
          if ((c.rank() == root) || all) resize_or_check_if_view(target_view, sha);
          if (std::abs(target_view.data() - source.data()) < source.size()) NDA_RUNTIME_ERROR << "mpi reduce of array : overlapping arrays !";
        }

        void *v_p       = (void *)target_view.data();
        void *rhs_p     = (void *)source.data();
        auto rhs_n_elem = source.size();
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
} // namespace nda::lazy_mpi

#if not(__cplusplus > 201703L)
namespace nda {
  //----------------------------  mark the class for C++17 concept workaround

  template <typename V, int R, uint64_t SO>
  inline constexpr bool is_array_initializer_v<lazy_mpi::reduce<V, R, SO>> = true;

} // namespace nda
#endif

//----------------------------

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
  AUTO(ArrayInitializer)
  mpi_reduce(A &a, mpi::communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) //
     REQUIRES(is_regular_or_view_v<std::decay_t<A>>) {
    //static_assert(has_layout_contiguous<std::decay_t<A>>, "Non contigous view in target_view.data() are not implemented");
    using v_t = std::decay_t<typename A::value_type>;
    using r_t = lazy_mpi::reduce<v_t, A::rank, A::layout_t::stride_order_encoded>;
    return r_t{typename r_t::view_t{a}, c, root, all, op};
  }

} // namespace nda
