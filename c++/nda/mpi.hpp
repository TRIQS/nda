#pragma once
#include <mpi/mpi.hpp>
#include "./mpi/reduce.hpp"
#include "./mpi/scatter.hpp"
#include "./mpi/gather.hpp"

namespace nda {

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

} // namespace nda
