// Copyright (c) 2020-2022 Simons Foundation
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

#include "./idx_map.hpp"
#include "./policies.hpp"

namespace nda {

  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  class rect_str;

  namespace details {
    // deduce rect_str from its base type
    template <typename T>
    struct rect_str_from_base;
    template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
    struct rect_str_from_base<idx_map<Rank, StaticExtents, StrideOrder, LayoutProp>> {
      using type = rect_str<Rank, StaticExtents, StrideOrder, LayoutProp>;
    };

  } // namespace details

  // -----------------------------------------------------------------------------------
  /**
   *
   * The layout that maps the indices to linear index, with additional string indices
   *
   */
  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  class rect_str : public idx_map<Rank, StaticExtents, StrideOrder, LayoutProp> {

    using ind_t  = nda::array<nda::array<std::string, 1>, 1>;
    using base_t = idx_map<Rank, StaticExtents, StrideOrder, LayoutProp>;
    mutable std::shared_ptr<ind_t const> s_indices; // the string indices for each dimension

    using base_t::n_dynamic_extents;

    static std::array<long, Rank> make_shape_from_string_indices(ind_t const &str_indices) {
      if (str_indices.size() != Rank) NDA_RUNTIME_ERROR << "String indices are not of the correct size";
      std::array<long, Rank> sha;
      for (int i = 0; i < Rank; ++i) sha[i] = str_indices[i].size();
      return sha;
    }

    public:
    // ----------------  Accessors -------------------------

    nda::array<nda::array<std::string, 1>, 1> const &get_string_indices() const {
      if (not s_indices) {
        auto ind = ind_t(Rank);
        for (int i = 0; i < Rank; ++i) {
          auto a = nda::array<std::string, 1>(this->lengths()[i]);
          for (int j = 0; j < a.size(); ++j) a(j) = std::to_string(j);
          ind(i) = std::move(a);
        }
        s_indices = std::make_shared<ind_t>(std::move(ind));
      }
      return *s_indices;
    }

    template <typename T>
    static constexpr int argument_is_allowed_for_call = base_t::template argument_is_allowed_for_call<T> or std::is_constructible_v<std::string, T>;

    template <typename T>
    static constexpr int argument_is_allowed_for_call_or_slice =
       base_t::template argument_is_allowed_for_call_or_slice<T> or std::is_constructible_v<std::string, T>;

    // ----------------  Constructors -------------------------

    /// Default constructor. Strides are not initiliazed.
    rect_str() = default;

    rect_str(rect_str const &) = default;
    rect_str(rect_str &&)      = default;
    rect_str &operator=(rect_str const &) = default;
    rect_str &operator=(rect_str &&) = default;

    ///
    rect_str(base_t const &idxm) noexcept : base_t{idxm} {}

    ///
    rect_str(base_t const &idxm, ind_t const &str_indices) noexcept : base_t{idxm}, s_indices{std::make_shared<ind_t>(std::move(str_indices))} {}

    ///
    template <layout_prop_e P>
    rect_str(rect_str<Rank, StaticExtents, StrideOrder, P> const &idxm) noexcept
       : base_t{idxm}, s_indices{std::make_shared<ind_t>(idxm.get_string_indices())} {}

    ///
    template <uint64_t SE, layout_prop_e P>
    rect_str(rect_str<Rank, SE, StrideOrder, P> const &idxm) noexcept(false)
       : base_t{idxm}, s_indices{std::make_shared<ind_t>(idxm.get_string_indices())} {}

    ///
    rect_str(std::array<long, Rank> const &shape, std::array<long, Rank> const &strides) noexcept : base_t{shape, strides} {}

    ///
    rect_str(std::array<long, Rank> const &shape) noexcept : base_t{shape} {}

    ///
    rect_str(nda::array<nda::array<std::string, 1>, 1> str_indices) noexcept
       : base_t{make_shape_from_string_indices(str_indices)}, s_indices{std::make_shared<ind_t>(std::move(str_indices))} {}

    ///
    rect_str(std::array<long, n_dynamic_extents> const &shape) noexcept requires((n_dynamic_extents != Rank) and (n_dynamic_extents != 0))
       : base_t{shape} {}

    // ----------------  Call operator -------------------------
    private:
    template <typename T>
    auto peel_string(int pos, T const &x) const {
      if constexpr (not std::is_constructible_v<std::string, T>)
        return x;
      else {
        auto const &sind = get_string_indices();
        auto const &idx  = sind[pos];
        auto it          = std::find(idx.begin(), idx.end(), x);
        if (it == idx.end())
          NDA_RUNTIME_ERROR << "Calling array with string. Key " << x << " at position " << pos << " does not match the indices " << sind;
        return it - idx.begin(); // 1d array, have LegacyRandomAccessIterator const iterator
      }
    }

    template <typename... Args, size_t... Is>
    [[nodiscard]] FORCEINLINE long call_impl(std::index_sequence<Is...>, Args... args) const {
      return base_t::operator()(peel_string(Is, args)...);
    }

    public:
    template <typename... Args>
    FORCEINLINE long operator()(Args const &... args) const {
      return call_impl(std::make_index_sequence<sizeof...(Args)>{}, args...);
    }

    // ----------------  Slice -------------------------

    private:
    template <typename... T, auto... Is>
    FORCEINLINE decltype(auto) slice_impl(std::index_sequence<Is...>, T const &... x) const {
      auto const [offset, idxm2] = base_t::slice(peel_string(Is, x)...);
      using new_rect_str_t       = typename details::rect_str_from_base<std::decay_t<decltype(idxm2)>>::type;

      if (not s_indices) return std::make_pair(offset, new_rect_str_t{idxm2});

      // slice the indices. Not optimized, but simple
      auto const &current_ind = get_string_indices();
      ind_t ind2((not argument_is_allowed_for_call<T> + ...)); // number of arg which are range, ellipsis, i.e. not arg of simple call

      auto l = [p = 0, &current_ind, &ind2](int n, auto const &y) mutable -> void {
        using U = std::decay_t<decltype(y)>;
        if constexpr (not argument_is_allowed_for_call<U>) { ind2[p++] = current_ind[n](y); }
      };
      (l(Is, x), ...);

      return std::make_pair(offset, new_rect_str_t{idxm2, ind2});
    }

    public:
    template <typename... Args>
    auto slice(Args const &... args) const {
      return slice_impl(std::make_index_sequence<sizeof...(args)>{}, args...);
    }

    // ----------------  Comparison -------------------------

    bool operator==(rect_str const &x) const { return base_t::operator==(x) and (!s_indices or !x.s_indices or (*s_indices == *(x.s_indices))); }
    bool operator!=(rect_str const &x) { return !(operator==(x)); }

    // ---------------- Transposition -------------------------

    template <uint64_t Permutation>
    auto transpose() const {

      auto idxm2 = base_t::template transpose<Permutation>();

      using new_rect_str_t = typename details::rect_str_from_base<std::decay_t<decltype(idxm2)>>::type;
      if (not s_indices) return new_rect_str_t{idxm2};

      ind_t ind2(s_indices->size());
      static constexpr std::array<int, Rank> permu = decode<Rank>(Permutation);
      for (int u = 0; u < Rank; ++u) { ind2[permu[u]] = (*s_indices)[u]; }
      return new_rect_str_t{idxm2, ind2};
    }
  };

  // -------------------- Policy ------------------

  struct C_stride_layout_str;
  struct F_stride_layout_str;

  struct C_layout_str {
    template <int Rank>
    using mapping = rect_str<Rank, 0, C_stride_order<Rank>, layout_prop_e::contiguous>;

    using with_lowest_guarantee_t = C_stride_layout_str;
    using contiguous_t            = C_layout_str;
  };

  struct F_layout_str {
    template <int Rank>
    using mapping = rect_str<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::contiguous>;

    using with_lowest_guarantee_t = F_stride_layout_str;
    using contiguous_t            = F_layout_str;
  };

  struct C_stride_layout_str {
    template <int Rank>
    using mapping = rect_str<Rank, 0, C_stride_order<Rank>, layout_prop_e::none>;

    using with_lowest_guarantee_t = C_stride_layout_str;
    using contiguous_t            = C_layout_str;
  };

  struct F_stride_layout_str {
    template <int Rank>
    using mapping = rect_str<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::none>;

    using with_lowest_guarantee_t = F_stride_layout_str;
    using contiguous_t            = F_layout_str;
  };

  template <uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  struct basic_layout_str {
    // FIXME C++20 : StrideOrder will be a std::array<int, Rank> WITH SAME rank
    template <int Rank>
    using mapping = rect_str<Rank, StaticExtents, StrideOrder, LayoutProp>;

    using with_lowest_guarantee_t = basic_layout_str<StaticExtents, StrideOrder, layout_prop_e::none>;
    using contiguous_t            = basic_layout_str<StaticExtents, StrideOrder, layout_prop_e::contiguous>;
  };

  namespace details {

    template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
    struct layout_to_policy<rect_str<Rank, StaticExtents, StrideOrder, LayoutProp>> {
      using type = basic_layout_str<StaticExtents, StrideOrder, LayoutProp>;
    };

    template <int Rank>
    struct layout_to_policy<rect_str<Rank, 0, C_stride_order<Rank>, layout_prop_e::contiguous>> {
      using type = C_layout_str;
    };

    template <int Rank>
    struct layout_to_policy<rect_str<Rank, 0, C_stride_order<Rank>, layout_prop_e::none>> {
      using type = C_stride_layout_str;
    };

  } // namespace details

} // namespace nda
