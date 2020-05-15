#pragma once

#include "group_indices.hpp"

namespace nda {

  // Regroup here some function which only transform the layout
  // to reinterpret the data
  // e.g. permuted_indices_view, regroup_indices, reshape, ...

  // First a general function that map any transform of the layout onto the basic_array_view

  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy, typename NewLayoutType>
  auto map_layout_transform(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> a, NewLayoutType const &new_layout) {
    using layout_policy = basic_layout<encode(NewLayoutType::static_extents), encode(NewLayoutType::stride_order), NewLayoutType::layout_prop>;
    return basic_array_view<T, NewLayoutType::rank(), layout_policy, Algebra, AccessorPolicy, OwningPolicy>{new_layout, a.storage()};
  }

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, typename NewLayoutType>
  auto map_layout_transform(basic_array<T, R, L, Algebra, ContainerPolicy> &&a, NewLayoutType const &new_layout) {
    using layout_policy = basic_layout<encode(NewLayoutType::static_extents), encode(NewLayoutType::stride_order), NewLayoutType::layout_prop>;
    return basic_array<T, NewLayoutType::rank(), layout_policy, Algebra, ContainerPolicy>{new_layout, std::move(a.storage())};
  }

  // ---------------  reshape ------------------------
  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, size_t R2>
  auto reshape(basic_array<T, R, L, Algebra, ContainerPolicy> &&a, std::array<long, R2> const &new_shape) {
    using idx_map_t = typename L::template mapping<R2>;
    EXPECTS_WITH_MESSAGE(a.size() == (std::accumulate(new_shape.cbegin(), new_shape.cend(), 1, std::multiplies<>{})),
                         "Reshape : the new shape has a incorrect number of elements");
    return map_layout_transform(std::move(a), idx_map_t{new_shape});
  }

  // for convenience, call it with std::array{1,2}.... Document ?
  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, size_t R2>
  auto reshape(basic_array<T, R, L, Algebra, ContainerPolicy> &&a, std::array<int, R2> const &new_shape) {
    return reshape(std::move(a), make_std_array<long>(new_shape));
  }

  // ---------------  reshaped_view ------------------------

  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy, size_t R2>
  auto reshaped_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> a, std::array<long, R2> const &new_shape) {
    using idx_map_t = typename L::template mapping<R2>;
    EXPECTS_WITH_MESSAGE(a.size() == (std::accumulate(new_shape.cbegin(), new_shape.cend(), 1, std::multiplies<>{})),
                         "Reshape : the new shape has a incorrect number of elements");
    EXPECTS_WITH_MESSAGE(a.indexmap().is_contiguous(), "reshaped_view only works with contiguous views");
    return map_layout_transform(a, idx_map_t{new_shape});
  }

  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy, size_t R2>
  auto reshaped_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> a, std::array<int, R2> const &new_shape) {
    return reshaped_view(a, make_std_array<long>(new_shape));
  }

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, size_t R2>
  auto reshaped_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a, std::array<long, R2> const &new_shape) {
    return reshaped_view(basic_array_view<T const, R, L, Algebra, default_accessor, borrowed>(a), new_shape);
  }

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, size_t R2>
  auto reshaped_view(basic_array<T, R, L, Algebra, ContainerPolicy> &a, std::array<long, R2> const &new_shape) {
    return reshaped_view(basic_array_view<T, R, L, Algebra, default_accessor, borrowed>(a), new_shape);
  }

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, size_t R2>
  auto reshaped_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a, std::array<int, R2> const &new_shape) {
    return reshaped_view(basic_array_view<T const, R, L, Algebra, default_accessor, borrowed>(a), new_shape);
  }

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, size_t R2>
  auto reshaped_view(basic_array<T, R, L, Algebra, ContainerPolicy> &a, std::array<int, R2> const &new_shape) {
    return reshaped_view(basic_array_view<T, R, L, Algebra, default_accessor, borrowed>(a), new_shape);
  }

  // --------------- permuted_indices_view------------------------

  template <ARRAY_INT Permutation, typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  auto permuted_indices_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> a) {
    return map_layout_transform(a, transpose<Permutation>(a.indexmap()));
  }

  template <ARRAY_INT Permutation, typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  auto permuted_indices_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
    return permuted_indices_view(basic_array_view<T const, R, L, Algebra, default_accessor, borrowed>(a));
  }

  template <ARRAY_INT Permutation, typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  auto permuted_indices_view(basic_array<T, R, L, Algebra, ContainerPolicy> &a) {
    return permuted_indices_view<Permutation>(basic_array_view<T, R, L, Algebra, default_accessor, borrowed>(a));
  }

  // ---------------  transpose ------------------------
  // for matrices ...
  template <typename A>
  auto transpose(A &&a) NDA_REQUIRES(is_regular_or_view_v<std::decay_t<A>> and (std::decay_t<A>::rank == 2)) {
    return permuted_indices_view<encode(std::array{1, 0})>(std::forward<A>(a));
  }

  // Transposed_view swap two indices
  template <int I, int J, typename A>
  auto transposed_view(A &&a) NDA_REQUIRES(is_regular_or_view_v<std::decay_t<A>>) {
    return permuted_indices_view<encode(permutations::transposition<std::decay_t<A>::rank>(I, J))>(std::forward<A>(a));
  }

  /// --------------- Grouping indices------------------------

  // FIXME : write the doc

  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy, typename... IntSequences>
  auto group_indices_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> a, IntSequences...) {
    return map_layout_transform(a, group_indices_layout(a.indexmap(), IntSequences{}...));
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
