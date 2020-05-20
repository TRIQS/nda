#pragma once
#include <mpi/mpi.hpp>

namespace nda {

  // Lazy reduce operation
  // FIXME : template ? T, R ?? Layout MUST be C ? or NOT ?
  template <typename A>
  struct lazy_mpi_reduce {

    A const &ref;        // the array in reference
    mpi::communicator c; // mpi comm
    const int root{};    //
    const bool all{};
    const MPI_Op op{};

    using value_type = typename A::value_type; // needed to construct array from this object (to pass requires on convertibility of types)

    /// compute the shape of the target array
    [[nodiscard]] auto shape() const { return ref.shape(); }

    ///
    template <typename View>
    void invoke(View &&v) const {

      static_assert(has_layout_contiguous<A>, "Non contigous view in mpi_broadcast are not implemented");

      auto rhs_n_elem = ref.size();
      auto D          = mpi::mpi_type<typename A::value_type>::get();

      bool in_place = (v.data_start() == ref.data_start());
      auto sha      = shape(); // WARNING : Keep this out of the if condition (shape USES MPI) !

      // some checks.
      if (in_place) {
        if (rhs_n_elem != v.size()) NDA_RUNTIME_ERROR << "mpi reduce of array : same pointer to data start, but different number of elements !";
      } else { // check no overlap
        if ((c.rank() == root) || all) resize_or_check_if_view(v, sha);
        if (std::abs(v.data_start() - ref.data_start()) < rhs_n_elem) NDA_RUNTIME_ERROR << "mpi reduce of array : overlapping arrays !";
      }

      void *v_p   = v.data_start();
      void *rhs_p = (void *)ref.data_start();

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
  };

  //----------------------------  mark the class as assignable to an array for array construction and array/array_view assignment -------------

#if not __cplusplus > 201703L

  template <typename A>
  inline constexpr bool is_array_initializer_v<lazy_mpi_reduce<A>> = true;

#endif

  //----------------------------

  template <typename A>
  lazy_mpi_reduce<A> mpi_reduce(A &a, mpi::communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) //
     REQUIRES(is_regular_or_view_v<A>) {

    static_assert(has_layout_contiguous<A>, "Non contigous view in mpi_broadcast are not implemented");
    static_assert(ArrayInitializer<lazy_mpi_reduce<A>>, "Internal");

    return {a, c, root, all, op};
  }

} // namespace nda
