// Copyright (c) 2019-2022 Simons Foundation
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

#include "group_indices.hpp"

namespace nda {

  // Regroup here some function which only transform the layout
  // to reinterpret the data
  // e.g. permuted_indices_view, regroup_indices, reshape, ...

  // First a general function that map any transform of the layout onto the basic_array_view
  // NB : Algebra is down to A is
  // FIXME : regroup + requires ? ?
  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy, typename NewLayoutType>
  auto map_layout_transform(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> a, NewLayoutType const &new_layout) {
    using layout_policy = typename details::layout_to_policy<NewLayoutType>::type;
    return basic_array_view<T, NewLayoutType::rank(), layout_policy, (NewLayoutType::rank() == R ? Algebra : 'A'), AccessorPolicy, OwningPolicy>{
       new_layout, a.storage()};
  }

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, typename NewLayoutType>
  auto map_layout_transform(basic_array<T, R, L, Algebra, ContainerPolicy> &&a, NewLayoutType const &new_layout) {
    using layout_policy = typename details::layout_to_policy<NewLayoutType>::type;
    return basic_array<T, NewLayoutType::rank(), layout_policy, (NewLayoutType::rank() == R ? Algebra : 'A'), ContainerPolicy>{
       new_layout, std::move(a.storage())};
  }

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, typename NewLayoutType>
  auto map_layout_transform(basic_array<T, R, L, Algebra, ContainerPolicy> &a, NewLayoutType const &new_layout) {
    using layout_policy = typename details::layout_to_policy<NewLayoutType>::type;
    return basic_array_view<T, NewLayoutType::rank(), layout_policy, (NewLayoutType::rank() == R ? Algebra : 'A')>{new_layout, a.storage()};
  }

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, typename NewLayoutType>
  auto map_layout_transform(basic_array<T, R, L, Algebra, ContainerPolicy> const &a, NewLayoutType const &new_layout) {
    using layout_policy = typename details::layout_to_policy<NewLayoutType>::type;
    return basic_array_view<T const, NewLayoutType::rank(), layout_policy, (NewLayoutType::rank() == R ? Algebra : 'A')>{new_layout,
                                                                                                                         std::move(a.storage())};
  }

  // ---------------  reshape ------------------------

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, std::integral Int, auto newRank>
  auto reshape(basic_array<T, R, L, Algebra, ContainerPolicy> &&a, std::array<Int, newRank> const &new_shape) {
    using layout_t = typename L::template mapping<newRank>;
    EXPECTS_WITH_MESSAGE(a.size() == (std::accumulate(new_shape.cbegin(), new_shape.cend(), 1, std::multiplies<>{})),
                         "Reshape : the new shape has a incorrect number of elements");
    return map_layout_transform(std::move(a), layout_t{new_shape});
  }

  // ---------------  reshaped_view ------------------------

  /// Reshape : contiguous view only [runtime checked]
  ///\param Int : shape are std::array<long, R> but the Int allows the user to pass int, or any integer and forget about it
  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy, //
            std::integral Int, auto newRank>

  auto reshaped_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> v, //
                     std::array<Int, newRank> const &new_shape) {

    using layout_t = typename L::template mapping<newRank>;
    EXPECTS_WITH_MESSAGE(v.size() == (std::accumulate(new_shape.cbegin(), new_shape.cend(), 1, std::multiplies<>{})),
                         "Reshape : the new shape has a incorrect number of elements");
    EXPECTS_WITH_MESSAGE(v.indexmap().is_contiguous(), "reshaped_view only works with contiguous views");
    return map_layout_transform(v, layout_t{stdutil::make_std_array<long>(new_shape)});
  }

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, std::integral Int, auto newRank>
  auto reshaped_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a, std::array<Int, newRank> const &new_shape) {
    return reshaped_view(basic_array_view<T const, R, L, Algebra, default_accessor, borrowed<mem::get_addr_space<decltype(a)>>>(a), new_shape);
  }

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, std::integral Int, auto newRank>
  auto reshaped_view(basic_array<T, R, L, Algebra, ContainerPolicy> &a, std::array<Int, newRank> const &new_shape) {
    return reshaped_view(basic_array_view<T, R, L, Algebra, default_accessor, borrowed<mem::get_addr_space<decltype(a)>>>(a), new_shape);
  }

  // --------------- permuted_indices_view------------------------

  template <ARRAY_INT Permutation, typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  auto permuted_indices_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> a) {
    return map_layout_transform(a, a.indexmap().template transpose<Permutation>());
  }

  template <ARRAY_INT Permutation, typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  auto permuted_indices_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a) {
    return permuted_indices_view<Permutation>(basic_array_view<T const, R, L, Algebra, default_accessor, borrowed<mem::get_addr_space<decltype(a)>>>(a));
  }

  template <ARRAY_INT Permutation, typename T, int R, typename L, char Algebra, typename ContainerPolicy>
  auto permuted_indices_view(basic_array<T, R, L, Algebra, ContainerPolicy> &a) {
    return permuted_indices_view<Permutation>(basic_array_view<T, R, L, Algebra, default_accessor, borrowed<mem::get_addr_space<decltype(a)>>>(a));
  }

  // ---------------  transpose ------------------------

  // FIXME : NAME

  // for matrices ...
  template <typename A>
  auto transpose(A &&a) requires(is_regular_or_view_v<A> and (get_rank<A> == 2)) {
    return permuted_indices_view<encode(std::array{1, 0})>(std::forward<A>(a));
  }

  // Transposed_view swap two indices
  template <int I, int J, typename A>
  auto transposed_view(A &&a) requires(is_regular_or_view_v<A>) {
    return permuted_indices_view<encode(permutations::transposition<get_rank<A>>(I, J))>(std::forward<A>(a));
  }

  // --------------- Grouping indices------------------------

  // FIXME : write the doc
  // FIXME : use "magnetic" placeholder

  template <typename T, int R, typename L, char Algebra, typename AccessorPolicy, typename OwningPolicy, typename... IntSequences>
  auto group_indices_view(basic_array_view<T, R, L, Algebra, AccessorPolicy, OwningPolicy> a, IntSequences...) {
    return map_layout_transform(a, group_indices_layout(a.indexmap(), IntSequences{}...));
  }

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, typename... IntSequences>
  auto group_indices_view(basic_array<T, R, L, Algebra, ContainerPolicy> const &a, IntSequences...) {
    return group_indices_view(basic_array_view<T const, R, L, Algebra, default_accessor, borrowed<mem::get_addr_space<decltype(a)>>>(a), IntSequences{}...);
  }

  template <typename T, int R, typename L, char Algebra, typename ContainerPolicy, typename... IntSequences>
  auto group_indices_view(basic_array<T, R, L, Algebra, ContainerPolicy> &a, IntSequences...) {
    return group_indices_view(basic_array_view<T, R, L, Algebra, default_accessor, borrowed<mem::get_addr_space<decltype(a)>>>(a), IntSequences{}...);
  }

  // --------------- Reinterpretation------------------------

  namespace impl {

    template <int N, auto R>
    constexpr std::array<int, R + N> complete_stride_order_with_fast(std::array<int, R> const &a) {
      auto r = stdutil::make_initialized_array<R + N>(0);
      for (int i = 0; i < R; ++i) r[i] = a[i];
      for (int i = 0; i < N; ++i) r[R + i] = R + i;
      return r;
    }
  } // namespace impl

  // Take an array or view and add N dimensions of size 1 in the fastest indices
  template <int N, typename A>
  auto reinterpret_add_fast_dims_of_size_one(A &&a) requires(nda::is_regular_or_view_v<A>) {

    auto const &lay = a.indexmap();
    using lay_t     = std::decay_t<decltype(lay)>;

    static constexpr uint64_t new_stride_order_encoded = encode(impl::complete_stride_order_with_fast<N>(lay_t::stride_order));
    // (lay_t::stride_order_encoded == 0 ? 0 : encode(impl::complete_stride_order_with_fast<N>(lay_t::stride_order)));

    static constexpr uint64_t new_static_extents_encoded = encode(stdutil::join(lay_t::static_extents, stdutil::make_initialized_array<N>(0)));
    //   (lay_t::static_extents_encoded == 0 ? 0 : encode(stdutil::join(lay_t::static_extents, stdutil::make_initialized_array<N>(0))));

    using new_lay_t = idx_map<get_rank<A> + N, new_static_extents_encoded, new_stride_order_encoded, lay_t::layout_prop>;

    auto shap1111 = stdutil::make_initialized_array<N>(1l);
    return map_layout_transform(std::forward<A>(a), new_lay_t{stdutil::join(lay.lengths(), shap1111), stdutil::join(lay.strides(), shap1111)});
  }

} // namespace nda
