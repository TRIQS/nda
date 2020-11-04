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

namespace nda {

  template <typename A>
  struct lazy_mpi_gather {

    A const &ref;        // the array in reference
    mpi::communicator c; // mpi comm
    const int root;      //
    const bool all;

    public:
    using value_type = typename A::value_type; // needed to construct array from this object (to pass requires on convertibility of types)

    /// compute the shape of the target array. WARNING : MAKES A MPI CALL.
    [[nodiscard]] auto shape() const {
      auto dims      = ref.shape();
      long slow_size = dims[0];
      if (!all) {
        dims[0] = mpi::reduce(slow_size, c, root); // valid only on root
        if (c.rank() != root) dims[0] = 1;         // valid only on root
      } else
        dims[0] = mpi::all_reduce(slow_size, c); // in this case, it is valid on all nodes

      return dims;
    }

    ///
    template <typename View>
    void invoke(View &&v) const {

      static_assert(has_layout_contiguous<A>, "Non contigous view in mpi_broadcast are not implemented");

      auto recvcounts = std::vector<int>(c.size());
      auto displs     = std::vector<int>(c.size() + 1, 0);
      int sendcount   = ref.size();
      auto D          = mpi::mpi_type<typename A::value_type>::get();

      auto sha = shape(); // WARNING : Keep this out of the if condition (shape USES MPI) !
      if (all || (c.rank() == root)) resize_or_check_if_view(v, sha);

      void *v_p         = v.data_start();
      const void *rhs_p = ref.data_start();

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
  //----------------------------  mark the class as assignable to an array for array construction and array/array_view assignment -------------

#if not(__cplusplus > 201703L)

  template <typename A>
  inline constexpr bool is_array_initializer_v<lazy_mpi_gather<A>> = true;

#endif

  //----------------------------

  template <typename A>
  AUTO(ArrayInitializer)
  mpi_gather(A &a, mpi::communicator c = {}, int root = 0, bool all = false) //
     REQUIRES(is_regular_or_view_v<A>) {

    static_assert(has_layout_contiguous<A>, "Non contigous view in mpi_broadcast are not implemented");

#if (__cplusplus > 201703L)
    static_assert(ArrayInitializer<lazy_mpi_gather<A>>, "Internal");
#endif

    return lazy_mpi_gather<A>{a, c, root, all};
  }
} // namespace nda
