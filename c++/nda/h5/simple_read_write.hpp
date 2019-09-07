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

  namespace h5_details {

    using v_t = h5::array_interface::v_t; // the proper vector type for the h5 interface

    // given the lengths and strides, return a L_tot. One function for all ranks (save code).
    // Assume layout is C.
    // use stride[rank -1]  =   strides_h5 [rank -1]
    //     stride[rank -2]  =   L[rank-1] * strides_h5 [rank -2]
    //     stride[rank -3]  =   L[rank-1] * L[rank-2] * strides_h5 [rank -3]
    //     stride[0]        =   L[rank-1] * L[rank-2] * L[1] * strides_h5 [0]
    //     size             =   L[rank-1] * L[rank-2] * L[1] * L[0]

    inline std::pair<v_t, v_t> get_L_tot_and_strides_h5(long const *stri, int rank, long total_size) {
      v_t Ltot(rank), strides_h5(rank);
     
      for (int u = 0; u < rank; ++u) strides_h5[u] = stri[u];

      long L0 = total_size;
      for (int u = rank - 2; u >= 0; --u) {
        // L[u+1] as  gcd of size and stride[u] ... stride[0]
        long L = L0;
        // L becomes the  gcd
        for (int v = u; v >= 0; --v) { L = std::gcd(L, strides_h5[v]); }
        // divides
        for (int v = u; v >= 0; --v) {
          strides_h5[v] /= L;
          L0 /= L;
        }
        Ltot[u + 1] = L;
      }
      Ltot[0] = L0;
      return {Ltot, strides_h5};
    }
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

    static constexpr bool is_complex = is_complex_v<typename A::value_t>;

    auto _get_ty = []() {
      if constexpr (!is_complex) {
        return h5::hdf5_type<get_value_t<A>>;
      } else {
        return h5::hdf5_type<typename get_value_t<A>::value_type>;
      }
    };

    h5::array_interface::h5_array_view v{_get_ty(), (void *)(a.data_start()), A::rank + (is_complex ? 1 : 0)};

    auto [L_tot, strides_h5] =  h5_details::get_L_tot_and_strides_h5(a.indexmap().strides().data(), A::rank, a.size());

    for (int u = 0; u < A::rank; ++u) {

      NDA_PRINT(u);
      NDA_PRINT(L_tot[u]);
      NDA_PRINT(strides_h5[u]);

      v.slab.count[u] = a.shape()[u];
      v.slab.stride[u] = strides_h5[u];
      v.L_tot[u]        = L_tot[u];
    }

    if (is_complex) {
      v.slab.count[A::rank] = 2; // stride is already one
      v.L_tot[A::rank]      = 2;
    }

    auto ds = h5::array_interface::write(g, name, v, true);

    if (is_complex) h5_write_attribute(ds, "__complex__", "1");
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

    if (lt.rank() != A::rank) NDA_RUNTIME_ERROR << " h5 read of nda::array : incorrect rank. In file: " << lt.rank() << "  In memory " << A::rank;
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

    h5::array_interface::h5_array_view v{h5::hdf5_type<get_value_t<A>>, (void *)(a.data_start()), A::rank + (is_complex ? 1 : 0)};
    for (int u = 0; u < A::rank; ++u) {
      v.slab.count[u] = L[u];
      v.L_tot[u]       = L[u];
    }
    if (is_complex) {
      v.slab.count[A::rank] = 2; // stride is already one
      v.L_tot[A::rank]      = 2;
    }

    h5::array_interface::read(g, name, v, lt);
  }


} // namespace nda
