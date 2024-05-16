// Copyright (c) 2019-2024 Simons Foundation
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
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

#pragma once

#include "./concepts.hpp"
#include "./declarations.hpp"
#include "./exceptions.hpp"
#include "./layout/for_each.hpp"
#include "./layout/range.hpp"
#include "./traits.hpp"

#include <h5/h5.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace nda {

  namespace detail {

    // Resize a given array to fit a given shape or check if a given view fits the shape.
    template <MemoryArray A>
    void resize_or_check(A &a, std::array<long, A::rank> const &shape) {
      if constexpr (is_regular_v<A>) {
        a.resize(shape);
      } else {
        if (a.shape() != shape) NDA_RUNTIME_ERROR << "Error in nda::detail::resize_or_check: Dimension mismatch: " << shape << " != " << a.shape();
      }
    }

    // Given an array/view, prepare and return the corresponding h5::array_view to be written/read into.
    template <MemoryArray A>
    auto prepare_h5_array_view(const A &a) {
      auto [parent_shape, h5_strides] = h5::array_interface::get_parent_shape_and_h5_strides(a.indexmap().strides().data(), A::rank, a.size());
      auto v = h5::array_interface::array_view{h5::hdf5_type<get_value_t<A>>(), (void *)a.data(), A::rank, is_complex_v<typename A::value_type>};
      for (int u = 0; u < A::rank; ++u) {
        v.slab.count[u]   = a.shape()[u];
        v.slab.stride[u]  = h5_strides[u];
        v.parent_shape[u] = parent_shape[u];
      }
      return v;
    }

    // Create an h5::char_buf from an 1-dimensional nda::MemoryArray.
    template <MemoryArrayOfRank<1> A>
    h5::char_buf to_char_buf(A const &a) {
      // get size of longest string
      size_t s = 0;
      for (auto &x : a) s = std::max(s, x.size() + 1);

      // copy each string to a buffer and pad with zeros
      std::vector<char> buf;
      buf.resize(a.size() * s, 0x00);
      size_t i = 0;
      for (auto &x : a) {
        strcpy(&buf[i * s], x.c_str());
        ++i;
      }

      // return h5::char_buf
      auto len = h5::v_t{size_t(a.size()), s};
      return {buf, len};
    }

    // Write an h5::char_buf into a 1-dimensional nda::MemoryArray.
    template <MemoryArrayOfRank<1> A>
    void from_char_buf(h5::char_buf const &cb, A &a) {
      // prepare the array
      resize_or_check(a, std::array<long, 1>{static_cast<long>(cb.lengths[0])});

      // loop over all strings
      auto len_string = cb.lengths[1];
      size_t i        = 0;
      for (auto &x : a) {
        x = "";
        x.append(&cb.buffer[i * len_string]);
        ++i;
      }
    }

  } // namespace detail

  /**
   * @brief Write an nda::MemoryArray to a new dataset/subgroup into an HDF5 file.
   *
   * @details The following arrays/views can be written to HDF5:
   * - 1-dimensional array/view of strings: It is first converted to an h5::char_buf before it is written.
   * - arbitrary array/view of scalars: The data is written directly into an h5::dataset with the same shape.
   * If the stride order is not C-order, the elements of the array/view are first copied into an array with
   * C-order layout.
   * - arbitrary array/view of some generic type: The elements are written one-by-one into a subgroup.
   *
   * @tparam A nda::MemoryArray type.
   * @param g h5::group in which the dataset/subgroup is created.
   * @param name Name of the dataset/subgroup to which the nda::MemoryArray is written.
   * @param a nda::MemoryArray to be written.
   * @param compress Whether to compress the dataset.
   */
  template <MemoryArray A>
  void h5_write(h5::group g, std::string const &name, A const &a, bool compress = true) {
    if constexpr (std::is_same_v<typename A::value_type, std::string>) {
      // 1-dimensional array/view of strings
      h5_write(g, name, detail::to_char_buf(a));
    } else if constexpr (is_scalar_v<typename A::value_type>) {
      // n-dimensional array/view of scalars
      // make a copy if the array/view is not in C-order and write the copy
      if (not a.indexmap().is_stride_order_C()) {
        using h5_arr_t  = nda::array<get_value_t<A>, A::rank>;
        auto a_c_layout = h5_arr_t{a.shape()};
        a_c_layout()    = a;
        h5_write(g, name, a_c_layout, compress);
        return;
      }

      // prepare the h5::array_view and write it
      auto v = detail::prepare_h5_array_view(a);
      h5::array_interface::write(g, name, v, compress);
    } else {
      // n-dimensional array/view of some generic unknown type
      auto g2 = g.create_group(name);
      h5_write(g2, "shape", a.shape());
      auto make_name = [](auto i0, auto... is) { return (std::to_string(i0) + ... + ("_" + std::to_string(is))); };
      nda::for_each(a.shape(), [&](auto... is) { h5_write(g2, make_name(is...), a(is...)); });
    }
  }

  /**
   * @brief Construct an h5::hyperslab and the corresponding shape from a given slice, i.e. a tuple containing integer,
   * nda::range, nda::range::all_t and nda::ellipsis objects.
   *
   * @details The hyperslab will have the same number of dimensions as the underlying dataspace, whereas the shape will
   * only contain the selected dimensions. For example, a 1-dimensional slice of a 3-dimensional dataset will return a
   * hyperslab with 3 dimensions and a shape with 1 dimension.
   *
   * @tparam NDim Number of selected dimensions (== Rank of the array/view to be written/read into).
   * @tparam IRs Types in the slice, i.e. integer, nda::range, nda::range::all_t or nda::ellipsis.
   * @param slice Tuple specifying the slice of the dataset to which the nda::MemoryArray is written.
   * @param ds_shape Shape of the underlying dataset.
   * @param is_complex Whether the data is complex valued.
   * @return A pair containing the hyperslab and the shape of the selected slice.
   */
  template <size_t NDim, typename... IRs>
  auto hyperslab_and_shape_from_slice(std::tuple<IRs...> const &slice, std::vector<h5::hsize_t> const &ds_shape, bool is_complex) {
    // number of parameters specifying the slice
    static constexpr auto size_of_slice = sizeof...(IRs);

    // number of nda::ellipsis objects in the slice (only 1 is allowed)
    static constexpr auto ellipsis_count = (std::is_same_v<IRs, ellipsis> + ... + 0);
    static_assert(ellipsis_count < 2, "Error in nda::hyperslab_and_shape_from_slice: Only a single ellipsis is allowed in slicing");
    static constexpr auto has_ellipsis = (ellipsis_count == 1);

    // position of the ellipsis in the slice parameters
    static constexpr auto ellipsis_position = [&]<size_t... Is>(std::index_sequence<Is...>) {
      if constexpr (has_ellipsis) return ((std::is_same_v<IRs, ellipsis> * Is) + ... + 0);
      return size_of_slice;
    }(std::index_sequence_for<IRs...>{});

    // number of integers in the slice parameters
    static constexpr auto integer_count = (std::integral<IRs> + ... + 0);

    // number of nda::range and nda::range::all_t objects in the slice parameters
    static constexpr auto range_count = size_of_slice - integer_count - ellipsis_count;
    static_assert((has_ellipsis && range_count <= NDim) || range_count == NDim,
                  "Error in nda::hyperslab_and_shape_from_slice: Rank does not match the number of non-trivial slice dimensions");

    // number of dimensions spanned by the ellipsis
    static constexpr auto ellipsis_width = NDim - range_count;

    // number of dimensions in the hyperslab
    static constexpr auto slab_rank = NDim + integer_count;

    // check rank of the dataset (must match the rank of the hyperslabs)
    auto ds_rank = ds_shape.size() - is_complex;
    if (slab_rank != ds_rank)
      NDA_RUNTIME_ERROR << "Error in nda::hyperslab_and_shape_from_slice: Incompatible dataset and slice ranks: " << ds_rank
                        << " != " << size_of_slice;

    // create and return the hyperslab and the shape array
    auto slab  = h5::array_interface::hyperslab(slab_rank, is_complex);
    auto shape = std::array<long, NDim>{};
    [&, m = 0]<size_t... Is>(std::index_sequence<Is...>) mutable {
      (
         [&]<typename IR>(size_t n, IR const &ir) mutable {
           if (n > ellipsis_position) n += (ellipsis_width - 1);
           if constexpr (std::integral<IR>) {
             slab.offset[n] = ir;
             slab.count[n]  = 1;
           } else if constexpr (std::is_same_v<IR, nda::ellipsis>) {
             for (auto k : range(n, n + ellipsis_width)) {
               slab.count[k] = ds_shape[k];
               shape[m++]    = ds_shape[k];
             }
           } else if constexpr (std::is_same_v<IR, nda::range>) {
             slab.offset[n] = ir.first();
             slab.stride[n] = ir.step();
             slab.count[n]  = ir.size();
             shape[m++]     = ir.size();
           } else {
             static_assert(std::is_same_v<IR, nda::range::all_t>);
             slab.count[n] = ds_shape[n];
             shape[m++]    = ds_shape[n];
           }
         }(Is, std::get<Is>(slice)),
         ...);
    }(std::make_index_sequence<size_of_slice>{});
    return std::make_pair(slab, shape);
  }

  /**
   * @brief Write an nda::MemoryArray into a slice (hyperslab) of an existing dataset in an HDF5 file.
   *
   * @details The hyperslab in the dataset is specified by providing a tuple of integer, nda::range, nda::range::all_t or
   * nda::ellipsis objects. It's shape must match the shape of the nda::MemoryArray to be written, otherwise an exception
   * is thrown.
   *
   * Only arrays/views with scalar types can be written to a slice.
   *
   * @tparam A nda::MemoryArray type.
   * @tparam IRs Types in the slice, i.e. integer, nda::range, nda::range::all_t or nda::ellipsis.
   * @param g h5::group which contains the dataset.
   * @param name Name of the dataset.
   * @param a nda::MemoryArray to be written.
   * @param slice Tuple specifying the slice of the dataset to which the nda::MemoryArray is written.
   */
  template <MemoryArray A, typename... IRs>
  void h5_write(h5::group g, std::string const &name, A const &a, std::tuple<IRs...> const &slice) {
    // compile-time checks
    constexpr int size_of_slice = sizeof...(IRs);
    static_assert(size_of_slice > 0, "Error in nda::h5_write: Invalid empty slice");
    static_assert(is_scalar_v<typename A::value_type>, "Error in nda::h5_write: Slicing is only supported for scalar types");
    static constexpr bool is_complex = is_complex_v<typename A::value_type>;

    // make a copy if the array/view is not in C-order and write the copy
    if (not a.indexmap().is_stride_order_C()) {
      using h5_arr_t  = nda::array<get_value_t<A>, A::rank>;
      auto a_c_layout = h5_arr_t{a.shape()};
      a_c_layout()    = a;
      h5_write(g, name, a_c_layout, slice);
      return;
    }

    // get dataset info and check that the datatypes of the dataset and the array match
    auto ds_info = h5::array_interface::get_dataset_info(g, name);
    if (is_complex != ds_info.has_complex_attribute)
      NDA_RUNTIME_ERROR << "Error in nda::h5_write: Dataset and array/view must both be complex or non-complex";

    // get hyperslab and shape from the slice and check that the shapes match
    auto const [sl, sh] = hyperslab_and_shape_from_slice<A::rank>(slice, ds_info.lengths, is_complex);
    if (sh != a.shape()) NDA_RUNTIME_ERROR << "Error in nda::h5_write: Incompatible slice and array shape: " << sh << " != " << a.shape();

    // prepare the h5::array_view and write it
    auto v = detail::prepare_h5_array_view(a);
    h5::array_interface::write_slice(g, name, v, sl);
  }

  /**
   * @brief Read into an nda::MemoryArray from an HDF5 file.
   *
   * @details The following arrays/views are supported:
   * - 1-dimensional arrays/views of strings: The data is read into an h5::char_buf and then copied into the array/view.
   * - arbitrary arrays/views of scalars: The data is read directly from an h5::dataset into the array/view. If the stride order
   * is not C-order, the data is first read into an array with C-order layout, before it is copied into the array/view. The data
   * to be read from the dataset can be restricted by providing a slice.
   * - arbitrary arrays/views of some generic type: The elements are read one-by-one from the subgroup.
   *
   * @note While array objects will always be resized to fit the shape of the data read from the file, views must have the same
   * shape as the data to be read. If this is not the case, an exception is thrown.
   *
   * @tparam A nda::MemoryArray type.
   * @tparam IRs Types in the slice, i.e. integer, nda::range, nda::range::all_t or nda::ellipsis.
   * @param g h5::group which contains the dataset/subgroup to read from.
   * @param name Name of the dataset/subgroup.
   * @param a nda::MemoryArray object to be read into.
   * @param slice Optional tuple specifying the slice of the dataset to be read.
   */
  template <MemoryArray A, typename... IRs>
  void h5_read(h5::group g, std::string const &name, A &a, std::tuple<IRs...> const &slice = {}) {
    constexpr bool slicing = (sizeof...(IRs) > 0);

    if constexpr (std::is_same_v<typename A::value_type, std::string>) {
      // 1-dimensional array/view of strings
      static_assert(!slicing, "Error in nda::h5_read: Slicing not supported for arrays/views of strings");
      h5::char_buf cb;
      h5_read(g, name, cb);
      detail::from_char_buf(cb, a);
    } else if constexpr (is_scalar_v<typename A::value_type>) {
      // n-dimensional array/view of scalars
      // read into a temporary array if the array/view is not in C-order and copy the elements
      if (not a.indexmap().is_stride_order_C()) {
        using h5_arr_t  = nda::array<typename A::value_type, A::rank>;
        auto a_c_layout = h5_arr_t{};
        h5_read(g, name, a_c_layout, slice);
        detail::resize_or_check(a, a_c_layout.shape());
        a() = a_c_layout;
        return;
      }

      // get dataset info
      auto ds_info = h5::array_interface::get_dataset_info(g, name);

      // allow to read non-complex data into a complex array
      static constexpr bool is_complex = is_complex_v<typename A::value_type>;
      if constexpr (is_complex) {
        if (!ds_info.has_complex_attribute) {
          array<double, A::rank> tmp;
          h5_read(g, name, tmp, slice);
          a = tmp;
          return;
        }
      }

      // get the hyperslab and the shape of the slice
      std::array<long, A::rank> shape{};
      auto slab = h5::array_interface::hyperslab{};
      if constexpr (slicing) {
        auto const [sl, sh] = hyperslab_and_shape_from_slice<A::rank>(slice, ds_info.lengths, is_complex);
        slab                = sl;
        shape               = sh;
      } else {
        for (int u = 0; u < A::rank; ++u) shape[u] = ds_info.lengths[u]; // NB : correct for complex
      }

      // resize the array or check that the shape matches
      detail::resize_or_check(a, shape);

      // check the rank of the dataset and the array
      auto ds_rank = ds_info.rank() - is_complex;
      if (!slicing && ds_rank != A::rank)
        NDA_RUNTIME_ERROR << "Error in nda::h5_read: Incompatible dataset and array ranks: " << ds_rank << " != " << A::rank;

      // prepare the h5::array_view and read into it
      auto v = detail::prepare_h5_array_view(a);
      h5::array_interface::read(g, name, v, slab);
    } else {
      // n-dimensional array/view of some generic unknown type
      static_assert(!slicing, "Error in nda::h5_read: Slicing not supported for arrays/views of generic types");
      auto g2 = g.open_group(name);

      // get and check the shape or resize the array
      std::array<long, A::rank> h5_shape;
      h5_read(g2, "shape", h5_shape);
      detail::resize_or_check(a, h5_shape);

      // read element-by-element using the appropriate h5_read implementation
      auto make_name = [](auto i0, auto... is) { return (std::to_string(i0) + ... + ("_" + std::to_string(is))); };
      nda::for_each(a.shape(), [&](auto... is) { h5_read(g2, make_name(is...), a(is...)); });
    }
  }

} // namespace nda
