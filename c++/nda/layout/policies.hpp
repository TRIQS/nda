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
//
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once
#include "idx_map.hpp"
#include "../concepts.hpp"

namespace nda {

  struct C_stride_layout;
  struct F_stride_layout;

  struct C_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, C_stride_order<Rank>, layout_prop_e::contiguous>;

    using with_lowest_guarantee_t = C_stride_layout;
    using contiguous_t            = C_layout;
  };

  struct F_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::contiguous>;

    using with_lowest_guarantee_t = F_stride_layout;
    using contiguous_t            = F_layout;
  };

  struct C_stride_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, C_stride_order<Rank>, layout_prop_e::none>;

    using with_lowest_guarantee_t = C_stride_layout;
    using contiguous_t            = C_layout;
  };

  struct F_stride_layout {
    template <int Rank>
    using mapping = idx_map<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::none>;

    using with_lowest_guarantee_t = F_stride_layout;
    using contiguous_t            = F_layout;
  };

  template <uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  struct basic_layout {
    // FIXME C++20 : StrideOrder will be a std::array<int, Rank> WITH SAME rank
    template <int Rank>
    using mapping = idx_map<Rank, StaticExtents, StrideOrder, LayoutProp>;

    using with_lowest_guarantee_t = basic_layout<StaticExtents, StrideOrder, layout_prop_e::none>;
    using contiguous_t            = basic_layout<StaticExtents, StrideOrder, layout_prop_e::contiguous>;
  };

  template <uint64_t StrideOrder>
  using contiguous_layout_with_stride_order = basic_layout<0, StrideOrder, layout_prop_e::contiguous>;

  template <int Rank, uint64_t StrideOrder>
  using get_contiguous_layout_policy =
     std::conditional_t<StrideOrder == C_stride_order<Rank>, C_layout,
                        std::conditional_t<StrideOrder == Fortran_stride_order<Rank>, F_layout, contiguous_layout_with_stride_order<StrideOrder>>>;

  namespace details {

    template <typename L>
    struct layout_to_policy;

    template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
    struct layout_to_policy<idx_map<Rank, StaticExtents, StrideOrder, LayoutProp>> {
      using type = basic_layout<StaticExtents, StrideOrder, LayoutProp>;
    };

    template <int Rank>
    struct layout_to_policy<idx_map<Rank, 0, C_stride_order<Rank>, layout_prop_e::contiguous>> {
      using type = C_layout;
    };

    template <int Rank>
    struct layout_to_policy<idx_map<Rank, 0, C_stride_order<Rank>, layout_prop_e::none>> {
      using type = C_stride_layout;
    };

    // NOT OK for Rank 1
    //template <int Rank>
    //struct layout_to_policy<idx_map<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::contiguous>> {
    //using type = F_layout;
    //};

    //template <int Rank>
    //struct layout_to_policy<idx_map<Rank, 0, Fortran_stride_order<Rank>, layout_prop_e::none>> {
    //using type = F_stride_layout;
    //};

  } // namespace details

} // namespace nda
