/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011-2014 by O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#include <h5/array_interface.hpp>

namespace nda {

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
    h5::details::h5_array_view v{h5::hdf5_type<get_value_t<A>>, (void *)(a.data_start()), A::rank};
    for (int u = 0; u < a.rank; ++u) v.slab.count[u] = a.shape()[u];
    h5::details::write(g, name, v, true);
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
    // First case : an array of basic type (int, ...);
    static_assert(std::is_same_v<std::string, get_value_t<A>> or is_scalar_v<get_value_t<A>>, "Only array on basic types or strings");

    auto lt = h5::details::get_h5_lengths_type(g, name);

    if (lt.lengths.size() != A::rank) NDA_RUNTIME_ERROR << " h5 read : incorrect rank";
    shape_t<A::rank> L;
    for (int u = 0; u < A::rank; ++u) L[u] = lt.lengths[u];

    if constexpr (is_regular_v<A>) {
      if (a.shape() != L) a.resize(L);
    } else {
      if (a.shape() != L)
        NDA_RUNTIME_ERROR << "Error trying to read from an hdf5 file to a view. Dimension mismatch"
                          << "\n in file  : " << L << "\n in view  : " << a.shape();
      if (!a.indexmap().is_contiguous()) NDA_RUNTIME_ERROR << " Non contiguous view : h5 read not implemented yet";
      // NOTION OF hyperslab vs nda strides ...
    }

    h5::details::h5_array_view v{h5::hdf5_type<get_value_t<A>>, (void *)(a.data_start()), A::rank};
    v.slab.count = lt.lengths;

    h5::details::read(g, name, v);
  }


} // namespace nda
