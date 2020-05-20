#pragma once
#include <mpi/mpi.hpp>

namespace nda {

  // mpi is implemented with a lazy mechanism for reduce, gather, scatter
  // a = mpi_reduce(b, ...)
  // mpi_reduce produces a lazy object, so that the call is rewritten
  // mpi_reduce( b, a, ....) using the direct C MPI.

  //--------------------------------------------------------------------------------------------------------
  // Store the lazy reduce
  //
  template <typename A>
  struct mpi_lazy_reduce {

    //static_assert(ArrayInitializer<mpi_lazy_reduce<A>>, "Internal");

    using value_type = typename A::value_type; // needed to construct array from this object (to pass requires on convertibility of types)

    A const &ref;        // the array in reference
    mpi::communicator c; // mpi comm
    int root{};          //
    bool all{};
    MPI_Op op{};

    /// compute the shape of the target array
    [[nodiscard]] auto shape() const { return ref.shape(); }

    ///
    template <typename View>
    void invoke(View &&v) const;

  };

  //--------------------------------------------------------------------------------------------------------
  // Store the lazy scatter
  //
  template <typename A>
  struct mpi_lazy_scatter {

    using value_type = typename A::value_type; // needed to construct array from this object (to pass requires on convertibility of types)

    A const &ref;        // the array in reference
    mpi::communicator c; // mpi comm
    int root{};          //
    bool all{};

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
    void invoke(View &&v) const;
  };

  //--------------------------------------------------------------------------------------------------------
  // Store the lazy gather
  //
  template <typename A>
  struct mpi_lazy_gather {

    using value_type = typename A::value_type; // needed to construct array from this object (to pass requires on convertibility of types)

    A const &ref;        // the array in reference
    mpi::communicator c; // mpi comm
    int root{};          //
    bool all{};

    /// compute the shape of the target array. WARNING : MAKES A MPI CALL.
    [[nodiscard]] auto shape() const {
      auto dims      = ref.shape();
      long slow_size = dims[0];
      if (!all) {
        dims[0] = mpi::reduce(slow_size, c, root); // valid only on root
        if (c.rank() != root) dims[0] = 1;         // valid only on root
      } else
        dims[0] = mpi::all_reduce(slow_size, c, root); // in this case, it is valid on all nodes

      return dims;
    }

    ///
    template <typename View>
    void invoke(View &&v) const;

  };

  //----------------------------  mark the class as assignable to an array for array construction and array/array_view assignment -------------

  template <typename A>
  inline constexpr bool is_array_initializer_v<mpi_lazy_reduce<A>> = true;

  template <typename A>
  inline constexpr bool is_array_initializer_v<mpi_lazy_gather<A>> = true;

  template <typename A>
  inline constexpr bool is_array_initializer_v<mpi_lazy_scatter<A>> = true;

  //--------------------------------------------------------------------------------------------------------
  // The function to call on the array
  //--------------------------------------------------------------------------------------------------------

  // FIXME : REQUIRE
  // NdArrayContainer or NdArrayView
  //
  template <typename A>
  void mpi_broadcast(A &a, mpi::communicator c = {}, int root = 0) //
     REQUIRES(is_regular_or_view_v<A>) {
    static_assert(has_layout_contiguous<A>, "Non contigous view in mpi_broadcast are not implemented");
    auto sh = a.shape();
    //FIXME : mpi::std::array
    MPI_Bcast(&sh[0], sh.size(), mpi::mpi_type<typename decltype(sh)::value_type>::get(), root, c.get());
    if (c.rank() != root) { resize_or_check_if_view(a, sh); }
    MPI_Bcast(a.data_start(), a.size(), mpi::mpi_type<typename A::value_type>::get(), root, c.get());
  }

  template <typename A>
  mpi_lazy_reduce<A> mpi_reduce(A &a, mpi::communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM)
     REQUIRES(is_regular_or_view_v<A>) {
    static_assert(has_layout_contiguous<A>, "Non contigous view in mpi_broadcast are not implemented");
    return {a, c, root, all, op};
  }

  template <typename A>
  mpi_lazy_scatter<A> mpi_scatter(A &a, mpi::communicator c = {}, int root = 0, bool all = false) //
     REQUIRES(is_regular_or_view_v<A>) {
    static_assert(has_layout_contiguous<A>, "Non contigous view in mpi_broadcast are not implemented");
    return {a, c, root, all};
  }

  template <typename A>
  mpi_lazy_gather<A> mpi_gather(A &a, mpi::communicator c = {}, int root = 0, bool all = false) //
     REQUIRES(is_regular_or_view_v<A>) {
    static_assert(has_layout_contiguous<A>, "Non contigous view in mpi_broadcast are not implemented");
    return {a, c, root, all};
  }

  //--------------------------------------------------------------------------------------------------------
  // Delegation of the assignment for the lazy object
  // a = mpi_lazy_whatever(...) --->  assign_from_lazy(a, mpi_lazy_...)
  //--------------------------------------------------------------------------------------------------------

  // --------------------- mpi_lazy_reduce -----------------------
  //
  template <typename A>
  template <typename View>
  void mpi_lazy_reduce<A>::invoke(View && v) const {

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

    void *v_p = v.data_start();
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

    PRINT(v);
  }

  // ----------------------- mpi_lazy_scatter ------------------------------------

  template <typename A>
  template <typename View>
  void mpi_lazy_scatter<A>::invoke(View && v) const {

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

  // ----------------------- mpi_lazy_gather ------------------------------------

  template <typename A>
  template <typename View>
  void mpi_lazy_gather<A>::invoke(View && v) const {
 
    static_assert(has_layout_contiguous<A>, "Non contigous view in mpi_broadcast are not implemented");

    auto recvcounts = std::vector<int>(c.size());
    auto displs     = std::vector<int>(c.size() + 1, 0);
    int sendcount   = ref.size();
    auto D          = mpi::mpi_type<typename A::value_type>::get();

    auto sha = shape(); // WARNING : Keep this out of the if condition (shape USES MPI) !
    if (all || (c.rank() == root)) resize_or_check_if_view(v, sha);

    void *v_p       = v.data_start();
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
} // namespace nda
