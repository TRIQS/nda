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

#include "traits.hpp"
#include "accessors.hpp"
#include "layout/policies.hpp"
#include "mem/policies.hpp"
#include "stdutil/array.hpp"

namespace nda {

  // ---------------------- The layout  --------------------------------

  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  class idx_map;

  // ---------------------- declare array and view  --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  class basic_array;

  template <typename ValueType, int Rank, typename Layout, char Algebra = 'A',
            typename AccessorPolicy = nda::default_accessor, //, nda::no_alias_accessor, //,
            typename OwningPolicy   = nda::borrowed<>>
  class basic_array_view;

  // ---------------------- User aliases  --------------------------------

  template <typename ValueType, int Rank, typename Layout = C_layout, typename ContainerPolicy = heap<>>
  using array = basic_array<ValueType, Rank, Layout, 'A', ContainerPolicy>;

  template <typename ValueType, int Rank, typename Layout = C_stride_layout>
  using array_view = basic_array_view<ValueType, Rank, Layout, 'A', default_accessor, borrowed<>>;

  template <typename ValueType, int Rank, typename Layout = C_stride_layout>
  using array_const_view = basic_array_view<ValueType const, Rank, Layout, 'A', default_accessor, borrowed<>>;

  template <typename ValueType, int Rank, typename Layout = C_layout>
  requires(has_contiguous(Layout::template mapping<Rank>::layout_prop)) using array_contiguous_view =
     basic_array_view<ValueType, Rank, Layout, 'A', default_accessor, borrowed<>>;

  template <typename ValueType, int Rank, typename Layout = C_layout>
  requires(has_contiguous(Layout::template mapping<Rank>::layout_prop)) using array_contiguous_const_view =
     basic_array_view<ValueType const, Rank, Layout, 'A', default_accessor, borrowed<>>;

  template <typename ValueType, typename Layout = C_layout, typename ContainerPolicy = heap<>>
  using matrix = basic_array<ValueType, 2, Layout, 'M', ContainerPolicy>;

  template <typename ValueType, typename Layout = C_stride_layout>
  using matrix_view = basic_array_view<ValueType, 2, Layout, 'M', default_accessor, borrowed<>>;

  template <typename ValueType, typename Layout = C_stride_layout>
  using matrix_const_view = basic_array_view<ValueType const, 2, Layout, 'M', default_accessor, borrowed<>>;

  template <typename ValueType, typename ContainerPolicy = heap<>>
  using vector = basic_array<ValueType, 1, C_layout, 'V', ContainerPolicy>;

  template <typename ValueType, typename Layout = C_stride_layout>
  using vector_view = basic_array_view<ValueType, 1, Layout, 'V', default_accessor, borrowed<>>;

  template <typename ValueType, typename Layout = C_stride_layout>
  using vector_const_view = basic_array_view<ValueType const, 1, Layout, 'V', default_accessor, borrowed<>>;

  template <typename ValueType, int Rank, uint64_t StaticExtents>
  using stack_array =
     nda::basic_array<ValueType, Rank, nda::basic_layout<StaticExtents, nda::C_stride_order<Rank>, nda::layout_prop_e::contiguous>, 'A', nda::stack<stdutil::product(decode<Rank>(StaticExtents))>>;

  template <typename... Is>
  constexpr uint64_t static_extents(int i0, Is... is) {
    if (i0 > 15) throw std::runtime_error("NO!");
    return encode(std::array<int, sizeof...(Is) + 1>{i0, is...});
  }

  // ---------------------- Cuda Aliases --------------------------------

  template <typename ValueType, int Rank, typename Layout = C_layout>
  using cuarray = basic_array<ValueType, Rank, Layout, 'A', heap<mem::Device>>;

  template <typename ValueType, int Rank, typename Layout = C_stride_layout>
  using cuarray_view = basic_array_view<ValueType, Rank, Layout, 'A', default_accessor, borrowed<mem::Device>>;

  template <typename ValueType, int Rank, typename Layout = C_stride_layout>
  using cuarray_const_view = basic_array_view<ValueType const, Rank, Layout, 'A', default_accessor, borrowed<mem::Device>>;

  template <typename ValueType, typename Layout = C_layout, typename ContainerPolicy = heap<mem::Device>>
  using cumatrix = basic_array<ValueType, 2, Layout, 'M', ContainerPolicy>;

  template <typename ValueType, typename Layout = C_stride_layout>
  using cumatrix_view = basic_array_view<ValueType, 2, Layout, 'M', default_accessor, borrowed<mem::Device>>;

  template <typename ValueType, typename Layout = C_stride_layout>
  using cumatrix_const_view = basic_array_view<ValueType const, 2, Layout, 'M', default_accessor, borrowed<mem::Device>>;

  template <typename ValueType>
  using cuvector = basic_array<ValueType, 1, C_layout, 'V', heap<mem::Device>>;

  template <typename ValueType, typename Layout = C_stride_layout>
  using cuvector_view = basic_array_view<ValueType, 1, Layout, 'V', default_accessor, borrowed<mem::Device>>;

  template <typename ValueType, typename Layout = C_stride_layout>
  using cuvector_const_view = basic_array_view<ValueType const, 1, Layout, 'V', default_accessor, borrowed<mem::Device>>;

  // ---------------------- is_array_or_view_container  --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  inline constexpr bool is_regular_v<basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>> = true;

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  inline constexpr bool is_view_v<basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>> = true;

  // ---------------------- algebra --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  inline constexpr char get_algebra<basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>> = Algebra;

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  inline constexpr char get_algebra<basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>> = Algebra;

  // ---------------------- get_layout_info --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  inline constexpr layout_info_t get_layout_info<basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>> =
     basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>::layout_t::layout_info;

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  inline constexpr layout_info_t get_layout_info<basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>> =
     basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>::layout_t::layout_info;

} // namespace nda
