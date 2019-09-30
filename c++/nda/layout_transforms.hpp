#pragma once

#include "group_indices.hpp"

namespace nda {

  // Regroup here some function which only transform the layout
  // to reinterpret the data
  // e.g. permuted_indices_view, regroup_indices, reshape, ...

  // First a general function that map any transform of the layout onto the basic_array_view

  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy, typename Transform>
  auto map_layout_transform(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> a, Transform transform) {
    auto new_idx_map    = transform(a.indexmap());
    using new_idx_map_t = decltype(new_idx_map);
    using layout_policy = basic_layout<encode(new_idx_map_t::static_extents), encode(new_idx_map_t::stride_order), new_idx_map_t::layout_prop>;
    return basic_array_view<T, new_idx_map_t::rank(), layout_policy, Algebra, AccessorPolicy, OwningPolicy>{new_idx_map, a.storage()};
  }

  // Useless ?
  //template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, typename Transform>
  //auto map_layout_transform(basic_array<T, R, L, Algebra, ContainerPolicy> const &a, Transform transform) {
  //return map_layout_transform(a(), transform);
  //}

  //template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, typename Transform>
  //auto map_layout_transform(basic_array<T, R, L, Algebra, ContainerPolicy> &a, Transform transform) {
  //return map_layout_transform(a(), transform);
  //}

  /// --------------- permuted_indices_view------------------------

  template <ARRAY_INT Permutation, typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  auto permuted_indices_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> a) {
    return map_layout_transform(a, [](auto &&x) { return transpose<Permutation>(x); });
  }

  template <ARRAY_INT Permutation, typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  auto permuted_indices_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
    return permuted_indices_view(basic_array_view<T const, R, L, Algebra, default_accessor, borrowed>(a));
  }

  template <ARRAY_INT Permutation, typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  auto permuted_indices_view(basic_array<T, R, L, Algebra, ContainerPolicy> &a) {
    return permuted_indices_view<Permutation>(basic_array_view<T, R, L, Algebra, default_accessor, borrowed>(a));
  }

  // for matrices ...
  template <typename A>
  auto transpose(A &&a) REQUIRES(is_regular_or_view_v<std::decay_t<A>> and (std::decay_t<A>::rank == 2)) {
    return permuted_indices_view<encode(std::array{1, 0})>(std::forward<A>(a));
  }

  // Transposed_view swap two indices
  template <int I, int J, typename A>
  auto transposed_view(A &&a) REQUIRES(is_regular_or_view_v<std::decay_t<A>>) {
    return permuted_indices_view<encode(permutations::transposition<std::decay_t<A>::rank>(I, J))>(std::forward<A>(a));
  }

  /// --------------- Grouping indices------------------------

  // FIXME : write the doc

  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy, typename... IntSequences>
  auto group_indices_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> a, IntSequences...) {
    return map_layout_transform(a, [](auto &&x) { return group_indices_layout(x, IntSequences{}...); });
  }

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, typename... IntSequences>
  auto group_indices_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a, IntSequences...) {
    return group_indices_view(basic_array_view<T const, R, L, Algebra, default_accessor, borrowed>(a), IntSequences{}...);
  }

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, typename... IntSequences>
  auto group_indices_view(basic_array<T, R, L, Algebra, ContainerPolicy> &a, IntSequences...) {
    return group_indices_view(basic_array_view<T, R, L, Algebra, default_accessor, borrowed>(a), IntSequences{}...);
  }

} // namespace nda
