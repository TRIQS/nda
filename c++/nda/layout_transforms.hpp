// Copyright (c) 2019-2023 Simons Foundation
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
#include "map.hpp"

namespace nda {

  // A general function that maps any transform of the layout onto the basic_array_view
  template <MemoryArray A, typename NewLayoutType>
  auto map_layout_transform(A &&a, NewLayoutType const &new_layout) {
    using A_t                     = std::remove_reference_t<A>;
    using layout_policy           = typename details::layout_to_policy<NewLayoutType>::type;
    static constexpr auto algebra = (NewLayoutType::rank() == get_rank<A> ? get_algebra<A> : 'A');
    if constexpr (is_regular_v<A> and !std::is_reference_v<A>) { // basic_array rvalue
      using container_policy_t = typename A_t::container_policy_t;
      return basic_array<typename A_t::value_type, NewLayoutType::rank(), layout_policy, algebra, container_policy_t>{new_layout,
                                                                                                                      std::forward<A>(a).storage()};
    } else {
      using value_t         = std::conditional_t<std::is_const_v<A_t>, const typename A_t::value_type, typename A_t::value_type>;
      using accessor_policy = typename get_view_t<A>::accessor_policy_t;
      using owning_policy   = typename get_view_t<A>::owning_policy_t;
      return basic_array_view<value_t, NewLayoutType::rank(), layout_policy, algebra, accessor_policy, owning_policy>{new_layout, a.storage()};
    }
  }

  // --------------- reshape ------------------------

  template <MemoryArray A, std::integral Int, auto newRank>
  auto reshape(A &&a, std::array<Int, newRank> const &new_shape) {
    using layout_t = typename std::decay_t<A>::layout_policy_t::template mapping<newRank>;
    EXPECTS_WITH_MESSAGE(a.size() == (std::accumulate(new_shape.cbegin(), new_shape.cend(), Int{1}, std::multiplies<>{})),
                         "Reshape : the new shape has a incorrect number of elements");
    EXPECTS_WITH_MESSAGE(a.indexmap().is_contiguous(), "reshape only works with contiguous data");
    return map_layout_transform(std::forward<A>(a), layout_t{stdutil::make_std_array<long>(new_shape)});
  }

  template <MemoryArray A, std::integral... Ints>
  auto reshape(A &&a, Ints... is) {
    return reshape(std::forward<A>(a), std::array<long, sizeof...(Ints)>{static_cast<long>(is)...});
  }

  template <MemoryArray A, std::integral Int, auto newRank>
  [[deprecated("Please use reshape(arr, shape) instead")]] auto reshaped_view(A &&a, std::array<Int, newRank> const &new_shape) {
    return reshape(std::forward<A>(a), new_shape);
  }

  // --------------- flatten ------------------------

  template <MemoryArray A>
  auto flatten(A &&a) {
    return reshape(std::forward<A>(a), std::array{a.size()});
  }

  // --------------- permuted_indices_view------------------------

  template <uint64_t Permutation, MemoryArray A>
  auto permuted_indices_view(A &&a) {
    return map_layout_transform(std::forward<A>(a), a.indexmap().template transpose<Permutation>());
  }

  // ---------------  transpose ------------------------

  /**
   * Perform transposition on the memory layout of the MemoryMatrix.
   * No operation is performed in memory.
   * For expression-calls with a single argument perform transposition
   * on the underlying array.
   *
   * @tparam The type of the array or expression
   * @param The array or expression
   * @return A view with reversed index order
   */
  template <typename A>
  auto transpose(A &&a)
    requires(MemoryMatrix<A> or is_instantiation_of_v<expr_call, A>)
  {
    if constexpr (MemoryMatrix<A>) {
      return permuted_indices_view<encode(std::array{1, 0})>(std::forward<A>(a));
    } else { // expr_call
      static_assert(std::tuple_size_v<decltype(a.a)> == 1, "Cannot transpose expr_call with more than one array argument");
      return map(a.f)(transpose(std::get<0>(std::forward<A>(a).a)));
    }
  }

  // Transposed_view swap two indices
  template <int I, int J, MemoryArray A>
  auto transposed_view(A &&a)
    requires(is_regular_or_view_v<A>)
  {
    return permuted_indices_view<encode(permutations::transposition<get_rank<A>>(I, J))>(std::forward<A>(a));
  }

  // --------------- Grouping indices------------------------

  // FIXME : write the doc
  // FIXME : use "magnetic" placeholder

  template <MemoryArray A, typename... IntSequences>
  auto group_indices_view(A &&a, IntSequences...) {
    return map_layout_transform(std::forward<A>(a), group_indices_layout(a.indexmap(), IntSequences{}...));
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
  auto reinterpret_add_fast_dims_of_size_one(A &&a)
    requires(nda::is_regular_or_view_v<A>)
  {

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
