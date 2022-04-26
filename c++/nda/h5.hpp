// Copyright (c) 2019-2021 Simons Foundation
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
//
// Authors: Olivier Parcollet, Nils Wentzell

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
  template <MemoryArray A>
  void h5_write(h5::group g, std::string const &name, A const &a) requires(is_regular_or_view_v<A>);

  /*
   * Read an array or a view from an hdf5 file
   * Allow to read only a slice of the dataset by providing an optional tuple of nda::range and/or integers
   * The HDF5 exceptions will be caught and rethrown as std::runtime_error
   *
   * @tparam A The type of the array/matrix/vector, etc..
   * @tparam IRs The types in the slice tuple, i.e. integer, index ranges, range::all_t or ellipsis
   * @param g The h5 group
   * @param a The array to be stored
   * @param slice Optional slice (tuple of nda::range and/or integer types) to limit which data to read
   */
  template <MemoryArray A, typename... IRs>
  void h5_read(h5::group g, std::string const &name, A &a, std::tuple<IRs...> const &slice = {}) requires(is_regular_or_view_v<A>);

  // ----- Implementation ------

  namespace h5_details {

    using h5::v_t;
    // in cpp to diminish template instantiations
    void write(h5::group g, std::string const &name, h5::datatype ty, void *start, int rank, bool is_complex, long const *lens, long const *strides,
               long total_size);

    // FIXME almost the same code as for vector. Factorize this ?
    // For the moment, 1d only : easy to implement, just change the construction of the lengths
    template <typename A>
    h5::char_buf to_char_buf(A const &v) requires(is_regular_or_view_v<A>) {
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
    void from_char_buf(h5::char_buf const &cb, A &v) requires(is_regular_or_view_v<A>) {
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

  template <MemoryArray A>
  void h5_write(h5::group g, std::string const &name, A const &a) requires(is_regular_or_view_v<A>) {

    // If array is not in C-order or not contiguous
    // copy into array with default layout and write
    if (not a.indexmap().is_stride_order_C() or not a.indexmap().is_contiguous()) {
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
      h5_details::write(g, name, h5::hdf5_type<get_value_t<A>>(), (void *)(a.data()), A::rank, is_complex, a.indexmap().lengths().data(),
                        a.indexmap().strides().data(), a.size());

    } else { // generic unknown type to hdf5
      auto g2 = g.create_group(name);
      h5_write(g2, "shape", a.shape());
      auto make_name = [](auto i0, auto... is) { return (std::to_string(i0) + ... + ("_" + std::to_string(is))); };
      nda::for_each(a.shape(), [&](auto... is) { h5_write(g2, make_name(is...), a(is...)); });
    }
  }

  /**
   * Create a h5 hyperslab and shape from a slice, i.e. a tuple containing types
   * nda::range, nda::range::all_t and integers
   *
   * The hyperslab will have the same number of dimensions as the length of the slice
   * The shape will only contain those dimensions of the slice that are not of integer type
   *
   * @param slice The slice tuple
   * @param lengths The dimensions of the underlying dataset
   * @param is_complex True iff the dataset holds complex values
   * @tparam NDim The number of dimensions in the shape
   * @tparam IRs The types in the slice tuple, i.e. integer, index ranges, range::all_t or ellipsis
   */
  template <size_t NDim, typename... IRs>
  auto hyperslab_and_shape_from_slice(std::tuple<IRs...> const &slice, std::vector<h5::hsize_t> const &lengths, bool is_complex) {

    static constexpr auto size_of_slice = sizeof...(IRs);

    static constexpr auto ellipsis_count = (std::is_same_v<IRs, ellipsis> + ... + 0);
    static_assert(ellipsis_count < 2, "Can only provide single ellipsis in slicing");
    static constexpr auto has_ellipsis      = (ellipsis_count == 1);
    static constexpr auto ellipsis_position = [&]<size_t... Is>(std::index_sequence<Is...>) {
      if constexpr (has_ellipsis) return ((std::is_same_v<IRs, ellipsis> * Is) + ... + 0);
      return size_of_slice;
    }
    (std::index_sequence_for<IRs...>{});

    static constexpr auto integer_count  = (std::integral<IRs> + ... + 0);
    static constexpr auto range_count    = size_of_slice - integer_count - ellipsis_count;
    static constexpr auto ellipsis_width = NDim - range_count;

    static_assert((has_ellipsis && range_count <= NDim) || range_count == NDim,
                  "Array rank does not match the number of non-trivial slice dimensions");

    static constexpr auto slab_rank = NDim + integer_count;
    auto rank_in_file               = lengths.size() - is_complex;
    if (slab_rank != rank_in_file)
      NDA_RUNTIME_ERROR << " h5 read of nda::array : incorrect slice rank. In file: " << rank_in_file << "  Rank of slice: " << size_of_slice;

    auto hsl   = h5::array_interface::hyperslab(slab_rank, is_complex);
    auto shape = std::array<long, NDim>{};
    [&, m = 0 ]<size_t... Is>(std::index_sequence<Is...>) mutable {
      (
         [&]<typename IR>(size_t n, IR const &ir) mutable {
           if (n > ellipsis_position) n += (ellipsis_width - 1);
           if constexpr (std::integral<IR>) {
             hsl.offset[n] = ir;
             hsl.count[n]  = 1;
           } else if constexpr (std::is_same_v<IR, nda::ellipsis>) {
             for (auto k : range(n, n + ellipsis_width)) {
               hsl.count[k] = lengths[k];
               shape[m++]   = lengths[k];
             }
           } else if constexpr (std::is_same_v<IR, nda::range>) {
             hsl.offset[n] = ir.first();
             hsl.stride[n] = ir.step();
             hsl.count[n]  = ir.size();
             shape[m++]    = ir.size();
           } else {
             static_assert(std::is_same_v<IR, nda::range::all_t>);
             hsl.count[n] = lengths[n];
             shape[m++]   = lengths[n];
           }
         }(Is, std::get<Is>(slice)),
         ...);
    }
    (std::make_index_sequence<size_of_slice>{});
    return std::make_pair(hsl, shape);
  }

  template <MemoryArray A, typename... IRs>
  void h5_read(h5::group g, std::string const &name, A &a, std::tuple<IRs...> const &slice) requires(is_regular_or_view_v<A>) {

    // If array is not in C-order or not contiguous
    // read into array with default layout and copy
    constexpr bool is_stride_order_C = std::decay_t<A>::layout_t::is_stride_order_C();
    static_assert(is_stride_order_C or is_regular_v<A>, "Cannot read into an array_view to an array with non C-style memory layout");
    if (not is_stride_order_C or not a.indexmap().is_contiguous()) {
      using h5_arr_t  = nda::array<typename A::value_type, A::rank>;
      auto a_c_layout = h5_arr_t{};
      h5_read(g, name, a_c_layout, slice);
      if constexpr (is_regular_v<A>) a.resize(a_c_layout.shape());
      a() = a_c_layout;
      return;
    }

    constexpr int size_of_slice = sizeof...(IRs);
    constexpr bool slicing      = (size_of_slice > 0);

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

      std::array<long, A::rank> shape;
      auto slice_slab = h5::array_interface::hyperslab{};

      if constexpr (slicing) {
        auto const [sl, sh] = hyperslab_and_shape_from_slice<A::rank>(slice, lt.lengths, is_complex);
        slice_slab = sl;
        shape      = sh;
      } else {
        for (int u = 0; u < A::rank; ++u) shape[u] = lt.lengths[u]; // NB : correct for complex
      }

      if constexpr (is_regular_v<A>) {
        a.resize(shape);
      } else {
        if (a.shape() != shape)
          NDA_RUNTIME_ERROR << "Error trying to read from an hdf5 file to a view. Dimension mismatch"
                            << "\n in file  : " << shape << "\n in view  : " << a.shape();
      }

      auto rank_in_file = lt.rank() - is_complex;
      h5::array_interface::h5_array_view v{h5::hdf5_type<get_value_t<A>>(), (void *)(a.data()), rank_in_file, is_complex};
      if constexpr (slicing) {
        v.slab.count = slice_slab.count;
        v.L_tot      = slice_slab.count;
      } else {
        for (int u = 0; u < A::rank; ++u) {
          v.slab.count[u] = shape[u];
          v.L_tot[u]      = shape[u];
        }
      }
      h5::array_interface::read(g, name, v, lt, slice_slab);

    } else { // generic unknown type to hdf5

      static_assert(!slicing, "Slicing not supported in generic h5_read");

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
