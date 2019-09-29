#pragma once

#include "traits.hpp"
#include "accessors.hpp"
#include "layout/policies.hpp"
#include "storage/policies.hpp"

namespace nda {

  // ---------------------- declare array and view  --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  class basic_array;

  template <typename ValueType, int Rank, typename Layout, char Algebra = 'A', typename AccessorPolicy = nda::default_accessor,
            typename OwningPolicy = nda::borrowed>
  class basic_array_view;

  // ---------------------- User aliases  --------------------------------

  template <typename ValueType, int Rank, typename Layout = C_layout>
  using array = basic_array<ValueType, Rank, Layout, 'A', heap>;

  template <typename ValueType, int Rank, typename Layout = C_stride_layout>
  using array_view = basic_array_view<ValueType, Rank, Layout, 'A', default_accessor, borrowed>;

  template <typename ValueType, int Rank, typename Layout = C_stride_layout>
  using array_const_view = basic_array_view<ValueType const, Rank, Layout, 'A', default_accessor, borrowed>;

  template <typename ValueType, typename Layout = C_layout> // CLayout or FLayout
  using matrix = basic_array<ValueType, 2, Layout, 'M', heap>;

  template <typename ValueType, typename Layout = C_stride_layout>
  using matrix_view = basic_array_view<ValueType, 2, Layout, 'M', default_accessor, borrowed>;

  template <typename ValueType, typename Layout = C_stride_layout>
  using matrix_const_view = basic_array_view<ValueType const, 2, Layout, 'M', default_accessor, borrowed>;

  //template <typename ValueType, typename Layout = C_layout> // CLayout or FLayout
  //using vector = basic_array<ValueType, 1, Layout, 'A', heap>;

  //template <typename ValueType, typename Layout = C_stride_layout>
  //using vector_view = basic_array_view<ValueType, 1, Layout, 'A', default_accessor, borrowed>;

  template <typename ValueType, int Rank, uint64_t StaticExtents>
  using stack_array =
     nda::basic_array<long, Rank, nda::basic_layout<StaticExtents, nda::C_stride_order<Rank>, nda::layout_prop_e::contiguous>, 'A', nda::stack>;

  template <typename... Is>
  constexpr uint64_t static_extents(int i0, Is... is) {
    return encode(std::array<int, sizeof...(Is) + 1>{i0, is...});
  }

  // ---------------------- is_array_or_view_container  --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  inline constexpr bool is_regular_v<basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>> = true;

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  inline constexpr bool is_view_v<basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>> = true;

  // ---------------------- concept  --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  inline constexpr bool is_ndarray_v<basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>> = true;

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  inline constexpr bool is_ndarray_v<basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>> = true;

  // ---------------------- algebra --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  inline constexpr char get_algebra<basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>> = Algebra;

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  inline constexpr char get_algebra<basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>> = Algebra;

  // ---------------------- get_layout_info --------------------------------

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  inline constexpr layout_info_t get_layout_info<basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>> =
     basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>::idx_map_t::layout_info;

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  inline constexpr layout_info_t get_layout_info<basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>> =
     basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>::idx_map_t::layout_info;

} // namespace nda
