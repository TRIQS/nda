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

      MPI_Scatterv((void *)ref.data_start(), &sendcounts[0], &displs[0], D, (void *)v.data_start(), recvcount, D, root, c.get());
    }
  };

  //----------------------------  mark the class as assignable to an array for array construction and array/array_view assignment -------------

#if not __cplusplus > 201703L

  template <typename A>
  inline constexpr bool is_array_initializer_v<lazy_mpi_scatter<A>> = true;

#endif

  template <typename A>
  lazy_mpi_scatter<A> mpi_scatter(A &a, mpi::communicator c = {}, int root = 0, bool all = false) //
     REQUIRES(is_regular_or_view_v<A>) {

    static_assert(ArrayInitializer<lazy_mpi_scatter<A>>, "Internal");
    static_assert(has_layout_contiguous<A>, "Non contigous view in mpi_broadcast are not implemented");

    return {a, c, root, all};
  }
} // namespace nda
