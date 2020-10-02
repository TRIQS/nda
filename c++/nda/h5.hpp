// Copyright (c) 2019-2020 Simons Foundation
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
#include <h5/array_interface.hpp>
#include <h5/stl/string.hpp>

#include "basic_array.hpp"
#include "exceptions.hpp"

namespace nda {

  /*
   * Write an array or a view into an hdf5 file
   * The HDF5 exceptions will be caught and rethrown as std::runtime_error
   *
   * @tparam A The type of the array/matrix/vector, etc..
   * @param g The h5 group
   * @param name The name of the hdf5 array in the file/group where the stack will be stored
   * @param a The array to be stored
   */
  template <typename A>
  void h5_write(h5::group g, std::string const &name, A const &a) REQUIRES(is_regular_or_view_v<A>);

  /*
   * Read an array or a view from an hdf5 file
   * The HDF5 exceptions will be caught and rethrown as std::runtime_error
   *
   * @tparam A The type of the array/matrix/vector, etc..
   * @param g The h5 group
   * @param a The array to be stored
   */
  template <typename A>
  void h5_read(h5::group g, std::string const &name, A &a) REQUIRES(is_regular_or_view_v<A>);

  // ----- Implementation ------

  namespace h5_details {

    using h5::v_t;
    // in cpp to diminish template instantiations
    void write(h5::group g, std::string const &name, h5::datatype ty, void *start, int rank, bool is_complex, long const *lens, long const *strides,
               long total_size);

    // FIXME almost the same code as for vector. Factorize this ?
    // For the moment, 1d only : easy to implement, just change the construction of the lengths
    template <typename A>
    h5::char_buf to_char_buf(A const &v) REQUIRES(is_regular_or_view_v<A>) {
      static_assert(A::rank == 1, "H5 for array<string, N> for N>1 not implemented");
      size_t s = 0;
      for (auto &x : v) s = std::max(s, x.size() + 1);
      auto len = v_t{size_t(v.size()), s};

      std::vector<char> buf;
      buf.resize(v.size() * s, 0x00);
      size_t i = 0;
      for (auto &x : v) {
        strcpy(&buf[i * s], x.c_str());
        ++i;
      }
      return {buf, len};
    }

    template <typename A>
    void from_char_buf(h5::char_buf const &cb, A &v) REQUIRES(is_regular_or_view_v<A>) {
      static_assert(A::rank == 1, "H5 for array<string, N> for N>1 not implemented");
      v.resize(cb.lengths[0]);
      auto len_string = cb.lengths[1];

      size_t i = 0;
      for (auto &x : v) {
        x = "";
        x.append(&cb.buffer[i * len_string]);
        ++i;
      }
    }

  } // namespace h5_details

  template <typename A>
  void h5_write(h5::group g, std::string const &name, A const &a) REQUIRES(is_regular_or_view_v<A>) {

    // Properly treat arrays with non-standard memory layout
    if constexpr (not std::decay_t<A>::layout_t::is_stride_order_C()) {
      using h5_arr_t  = nda::array<typename A::value_type, A::rank>;
      auto a_c_layout = h5_arr_t{a.shape()};
      a_c_layout()    = a;
      h5_write(g, name, a_c_layout);
      return;
    }

    // first case array of string
    if constexpr (std::is_same_v<typename A::value_type, std::string>) { // special case of string. Like vector of string

      h5_write(g, name, h5_details::to_char_buf(a));

    } else if constexpr (is_scalar_v<typename A::value_type>) { // FIXME : register types as USER DEFINED hdf5 types

      static constexpr bool is_complex = is_complex_v<typename A::value_type>;
      h5_details::write(g, name, h5::hdf5_type<get_value_t<A>>(), (void *)(a.data_start()), A::rank, is_complex, a.indexmap().lengths().data(),
                        a.indexmap().strides().data(), a.size());

    } else { // generic unknown type to hdf5
      auto g2 = g.create_group(name);
      h5_write(g2, "shape", a.shape());
      auto make_name = [](auto i0, auto... is) { return (std::to_string(i0) + ... + ("_" + std::to_string(is))); };
      nda::for_each(a.shape(), [&](auto... is) { h5_write(g2, make_name(is...), a(is...)); });
    }
  }

  template <typename A>
  void h5_read(h5::group g, std::string const &name, A &a) REQUIRES(is_regular_or_view_v<A>) {

    // If array is not C-strided, read into array with default layout and copy
    if constexpr (not std::decay_t<A>::layout_t::is_stride_order_C()) {
      static_assert(is_regular_v<A>, "Cannot read into an array_view to an array with non C-style memory layout");
      using h5_arr_t  = nda::array<typename A::value_type, A::rank>;
      auto a_c_layout = h5_arr_t{};
      h5_read(g, name, a_c_layout);
      a.resize(a_c_layout.shape());
      a() = a_c_layout;
      return;
    }

    // Special case array<string>, store as char buffer
    if constexpr (std::is_same_v<typename A::value_type, std::string>) {
      h5::char_buf cb;
      h5_read(g, name, cb);
      h5_details::from_char_buf(cb, a);

    } else if constexpr (is_scalar_v<typename A::value_type>) { // FIXME : register types as USER DEFINED hdf5 types

      static constexpr bool is_complex = is_complex_v<typename A::value_type>;

      auto lt = h5::array_interface::get_h5_lengths_type(g, name);

      // Allow to read non-complex data into array<complex>
      if (is_complex && !lt.has_complex_attribute) {
        array<double, A::rank> tmp;
        h5_read(g, name, tmp);
        a = tmp;
        return;
      }

      int rank_in_file = lt.rank() - (is_complex ? 1 : 0);
      if (rank_in_file != A::rank)
        NDA_RUNTIME_ERROR << " h5 read of nda::array : incorrect rank. In file: " << rank_in_file << "  In memory " << A::rank;
      std::array<long, A::rank> L;
      for (int u = 0; u < A::rank; ++u) L[u] = lt.lengths[u]; // NB : correct for complex

      if constexpr (is_regular_v<A>) {
        a.resize(L);
      } else {
        if (a.shape() != L)
          NDA_RUNTIME_ERROR << "Error trying to read from an hdf5 file to a view. Dimension mismatch"
                            << "\n in file  : " << L << "\n in view  : " << a.shape();
      }

      h5::array_interface::h5_array_view v{h5::hdf5_type<get_value_t<A>>(), (void *)(a.data_start()), A::rank, is_complex};
      for (int u = 0; u < A::rank; ++u) {
        v.slab.count[u] = L[u];
        v.L_tot[u]      = L[u];
      }
      h5::array_interface::read(g, name, v, lt);

    } else { // generic unknown type to hdf5
      auto g2 = g.open_group(name);

      // Reshape if necessary
      std::array<long, A::rank> h5_shape;
      h5_read(g2, "shape", h5_shape);
      if (a.shape() != h5_shape) a.resize(h5_shape);

      // Read using appropriate h5_read implementation
      auto make_name = [](auto i0, auto... is) { return (std::to_string(i0) + ... + ("_" + std::to_string(is))); };
      nda::for_each(a.shape(), [&](auto... is) { h5_read(g2, make_name(is...), a(is...)); });
    }
  }

} // namespace nda
