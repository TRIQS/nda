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

  //--------------------------------------------------------------------------------------------------------
  // Store the lazy scatter
  //
  template <typename A>
  struct lazy_mpi_scatter {

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
      mpi::broadcast(slow_size, c, root);
      dims[0] = mpi::chunk_length(slow_size, c.size(), c.rank());
      return dims;
    }

    ///
    template <typename View>
    void invoke(View &&v) const {

      static_assert(has_layout_contiguous<A>, "Non contigous view in mpi_broadcast are not implemented");

      auto sha = shape(); // WARNING : Keep this out of any if condition (shape USES MPI) !
      resize_or_check_if_view(v, sha);

      auto slow_size   = ref.extent(0);
      auto slow_stride = ref.indexmap().strides()[0];
      auto sendcounts  = std::vector<int>(c.size());
      auto displs      = std::vector<int>(c.size() + 1, 0);
      int recvcount    = mpi::chunk_length(slow_size, c.size(), c.rank()) * slow_stride;
      auto D           = mpi::mpi_type<typename A::value_type>::get();

      for (int r = 0; r < c.size(); ++r) {
        sendcounts[r] = mpi::chunk_length(slow_size, c.size(), r) * slow_stride;
        displs[r + 1] = sendcounts[r] + displs[r];
      }

      MPI_Scatterv((void *)ref.data(), &sendcounts[0], &displs[0], D, (void *)v.data(), recvcount, D, root, c.get());
    }
  };

  //----------------------------  mark the class as assignable to an array for array construction and array/array_view assignment -------------

#if not(__cplusplus > 201703L)

  template <typename A>
  inline constexpr bool is_array_initializer_v<lazy_mpi_scatter<A>> = true;

#endif

  template <typename A>
  AUTO(ArrayInitializer)
  mpi_scatter(A &a, mpi::communicator c = {}, int root = 0, bool all = false) //
     REQUIRES(is_regular_or_view_v<A>) {

    static_assert(has_layout_contiguous<A>, "Non contigous view in mpi_broadcast are not implemented");

#if (__cplusplus > 201703L)
    static_assert(ArrayInitializer<lazy_mpi_scatter<A>>, "Internal");
#endif

    return lazy_mpi_scatter<A>{a, c, root, all};
  }
} // namespace nda
