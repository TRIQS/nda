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
#include <ostream>

// ---------------------------------------------------------------

namespace nda {

  inline std::ostream &operator<<(std::ostream &out, layout_prop_e p) {
    return out << (has_contiguous(p) ? "contiguous   " : " ") << (has_strided_1d(p) ? "strided_1d   " : " ")
               << (has_smallest_stride_is_one(p) ? "smallest_stride_is_one   " : " ");
  }

  // idx_map
  template <int Rank, uint64_t StaticExtents, uint64_t StrideOrder, layout_prop_e LayoutProp>
  std::ostream &operator<<(std::ostream &out, idx_map<Rank, StaticExtents, StrideOrder, LayoutProp> const &x) {
    return out << "  Lengths  : " << x.lengths() << "\n"
               << "  Strides  : " << x.strides() << "\n"
               << "  StaticExtents  : " << decode<Rank>(StaticExtents) << "\n"
               << "  MemoryStrideOrder   : " << x.stride_order << "\n"
               << "  Flags   :  " << LayoutProp << "\n";
  }

  // ==============================================

  // array
  template <typename A>
  std::ostream &operator<<(std::ostream &out, A const &a) requires(is_regular_or_view_v<A>) {

    if constexpr (A::rank == 1) {
      out << "[";
      auto const &len = a.indexmap().lengths();
      for (size_t i = 0; i < len[0]; ++i) out << (i > 0 ? "," : "") << a(i);
      out << "]";
    }

    if constexpr (A::rank == 2) {
      auto const &len = a.indexmap().lengths();
      out << "\n[";
      for (size_t i = 0; i < len[0]; ++i) {
        out << (i == 0 ? "[" : " [");
        for (size_t j = 0; j < len[1]; ++j) out << (j > 0 ? "," : "") << a(i, j);
        out << "]" << (i == len[0] - 1 ? "" : "\n");
      }
      out << "]";
    }

    // FIXME : not very pretty, do better here, but that was the arrays way
    if constexpr (A::rank > 2) {
      out << "[";
      for (size_t j = 0; j < a.size(); ++j) out << (j > 0 ? "," : "") << a.data()[j];
      out << "]";
    }

    return out;
  }

  // ==============================================

  template <int R, typename F>
  std::ostream &operator<<(std::ostream &sout, array_adapter<R, F> const &x) {
    return sout << "array_adapter of shape" << x.shape();
  }

  // ==============================================

  template <char OP, Array A> struct expr_unary; // Necessary for libclang parsing
  template <char OP, Array A>
  std::ostream &operator<<(std::ostream &sout, expr_unary<OP, A> const &x) {
    return sout << OP << x.a;
  }

  template <char OP, ArrayOrScalar L, ArrayOrScalar R> struct expr; // Necessary for libclang parsing
  template <char OP, ArrayOrScalar L, ArrayOrScalar R>
  std::ostream &operator<<(std::ostream &sout, expr<OP, L, R> const &x) {
    return sout << "(" << x.l << " " << OP << " " << x.r << ")";
  }

  // ==============================================

  template <typename F, typename... A>
  std::ostream &operator<<(std::ostream &out, expr_call<F, A...> const &) {
    return out << "mapped"; //array<value_type, std::decay_t<A>::rank>(x);
  }

} // namespace nda
