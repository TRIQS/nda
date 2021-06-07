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

// Models ArrayInitializer concept
template <nda::Array A>
REQUIRES17(nda::is_ndarray_v<std::decay_t<A>>)
struct mpi::lazy<mpi::tag::gather, A> {

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
    if (!all) {
      dims[0] = mpi::reduce(slow_size, c, root); // valid only on root
      if (c.rank() != root) dims[0] = 1;         // valid only on root
    } else
      dims[0] = mpi::all_reduce(slow_size, c); // in this case, it is valid on all nodes

    return dims;
  }

  /// Execute the mpi operation and write result to target
  template <nda::Array T>
  REQUIRES17(nda::is_ndarray_v<std::decay_t<T>>)
  void invoke(T &&target) const {
    if (not target.is_contiguous()) NDA_RUNTIME_ERROR << "mpi operations require contiguous target.data() to be contiguous";

    static_assert(std::decay_t<A>::layout_t::stride_order_encoded == std::decay_t<T>::layout_t::stride_order_encoded,
                  "Array types for rhs and target have incompatible stride order");

    if (not mpi::has_env) {
      target = rhs;
      return;
    }

    auto recvcounts = std::vector<int>(c.size());
    auto displs     = std::vector<int>(c.size() + 1, 0);
    int sendcount   = rhs.size();
    auto D          = mpi::mpi_type<value_type>::get();

    auto sha = shape(); // WARNING : Keep this out of the if condition (shape USES MPI) !
    if (all || (c.rank() == root)) resize_or_check_if_view(target, sha);

    void *v_p         = target.data();
    const void *rhs_p = rhs.data();

    auto mpi_ty = mpi::mpi_type<int>::get();
    if (!all)
      MPI_Gather(&sendcount, 1, mpi_ty, &recvcounts[0], 1, mpi_ty, root, c.get());
    else
      MPI_Allgather(&sendcount, 1, mpi_ty, &recvcounts[0], 1, mpi_ty, c.get());

    for (int r = 0; r < c.size(); ++r) displs[r + 1] = recvcounts[r] + displs[r];

    if (!all)
      MPI_Gatherv((void *)rhs_p, sendcount, D, v_p, &recvcounts[0], &displs[0], D, root, c.get());
    else
      MPI_Allgatherv((void *)rhs_p, sendcount, D, v_p, &recvcounts[0], &displs[0], D, c.get());
  }
};

namespace nda {

#if not(__cplusplus > 201703L)
  //----------------------------  mark the class for C++17 concept workaround
  template <typename A>
  REQUIRES17(nda::is_ndarray_v<std::decay_t<A>>)
  inline constexpr bool is_array_initializer_v<mpi::lazy<mpi::tag::gather, A>> = true;
#endif

  /**
   * Gather the array from mpi threads
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
  AUTO(ArrayInitializer)
  mpi_gather(A &&a, mpi::communicator c = {}, int root = 0, bool all = false) REQUIRES(is_regular_or_view_v<std::decay_t<A>>) {

    if (not a.is_contiguous()) NDA_RUNTIME_ERROR << "mpi operations require contiguous rhs.data() to be contiguous";

    return mpi::lazy<mpi::tag::gather, A>{std::forward<A>(a), c, root, all};
  }

} // namespace nda
