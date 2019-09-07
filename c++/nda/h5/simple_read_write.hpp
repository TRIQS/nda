#pragma once
#include <h5/array_interface.hpp>

namespace nda {

  namespace h5_details {

    // in cpp to diminish template instantiations
    void write(h5::group g, std::string const &name, h5::datatype ty, void *start, int rank, bool is_complex, long const *lens, long const *strides,
               long total_size);

  } // namespace h5_details

  /*
   * Write an array or a view into an hdf5 file
   * ArrayType The type of the array/matrix/vector, etc..
   * g The h5 group
   * name The name of the hdf5 array in the file/group where the stack will be stored
   * A The array to be stored
   * The HDF5 exceptions will be caught and rethrown as TRIQS_RUNTIME_ERROR (with a full stackstrace, cf triqs doc).
   */
  template <typename A>
  void h5_write(h5::group g, std::string const &name, A const &a) REQUIRES(is_regular_or_view_v<A>) {
    static_assert(std::is_same_v<std::string, get_value_t<A>> or is_scalar_v<get_value_t<A>>, "Only array on basic types or strings");

    if constexpr (std::is_same_v<typename A::value_t, std::string>) { // special case of string. Like vector of string
//      h5_write(g, name, to_char_buf(a))); 
    } 
    static constexpr bool is_complex = is_complex_v<typename A::value_t>;

    h5_details::write(g, name, h5::hdf5_type<get_value_t<A>>, (void *)(a.data_start()), A::rank, is_complex, a.indexmap().lengths().data(), a.indexmap().strides().data(),
                      a.size());
  }

  /*
   * Read an array or a view from an hdf5 file
   * ArrayType The type of the array/matrix/vector, etc..
   * g The h5 group
   * name The name of the hdf5 array in the file/group where the stack will be stored
   * A The array to be stored
   * The HDF5 exceptions will be caught and rethrown as std::runtime_error (with a full stackstrace, cf doc).
   */
  template <typename A>
  void h5_read(h5::group g, std::string const &name, A &a) REQUIRES(is_regular_or_view_v<A>) {

    static_assert(std::decay_t<decltype(a.indexmap())>::is_layout_C(), "Not implemented");
    static_assert(std::is_same_v<std::string, get_value_t<A>> or is_scalar_v<get_value_t<A>>, "Only array on basic types or strings");

    auto lt = h5::array_interface::get_h5_lengths_type(g, name);

    static constexpr bool is_complex = is_complex_v<typename A::value_t>;

    int rank_in_file = lt.rank() - (is_complex ? 1 : 0);
    if (rank_in_file != A::rank)
      NDA_RUNTIME_ERROR << " h5 read of nda::array : incorrect rank. In file: " << rank_in_file << "  In memory " << A::rank;
    shape_t<A::rank> L;
    for (int u = 0; u < A::rank; ++u) L[u] = lt.lengths[u]; // NB : correct for complex

    if constexpr (is_regular_v<A>) {
      if (a.shape() != L) a.resize(L);
    } else {
      if (a.shape() != L)
        NDA_RUNTIME_ERROR << "Error trying to read from an hdf5 file to a view. Dimension mismatch"
                          << "\n in file  : " << L << "\n in view  : " << a.shape();
    }

    h5::array_interface::h5_array_view v{h5::hdf5_type<get_value_t<A>>, (void *)(a.data_start()), A::rank, is_complex};
    for (int u = 0; u < A::rank; ++u) {
      v.slab.count[u] = L[u];
      v.L_tot[u]      = L[u];
    }

    h5::array_interface::read(g, name, v, lt);
  }

   

} // namespace nda
