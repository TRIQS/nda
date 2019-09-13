#pragma once
#include <ostream>

// FIXME : MOVE THIS
namespace std { // For ADL
  template <typename T, size_t R>
  std::ostream &operator<<(std::ostream &out, std::array<T, R> const &a) {
    return out << to_string(a);
  }

  template <typename T, size_t R>
  std::string to_string(std::array<T, R> const &a) {
    std::stringstream fs;
    fs << "(";
    for (int i = 0; i < R; ++i) fs << (i == 0 ? "" : " ") << a[i];
    fs << ")";
    return fs.str();
  }
} // namespace std

// ---------------------------------------------------------------

namespace nda {

  // idx_map
  template <int Rank, uint64_t StrideOrder, layout_info_e LayoutInfo>
  std::ostream &operator<<(std::ostream &out, idx_map<Rank, StrideOrder, LayoutInfo> const &x) {
    return out << "  Lengths  : " << x.lengths() << "\n"
               << "  Strides  : " << x.strides() << "\n"
               << "  MemoryStrideOrder   : " << x.stride_order << "\n"
               << "  Flags   :  " << (LayoutInfo & layout_info_e::contiguous ? "contiguous   " : " ") << (LayoutInfo & layout_info_e::strided_1d ? "strided_1d   " : " ")
               << (LayoutInfo & layout_info_e::smallest_stride_is_one ? "smallest_stride_is_one   " : " " )<< "\n";
  }

  // ==============================================

  // array
  template <typename A>
  std::ostream &operator<<(std::ostream &out, A const &a) REQUIRES(is_regular_or_view_v<A>) {

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
      for (size_t j = 0; j < a.size(); ++j) out << (j > 0 ? "," : "") << a.data_start()[j];
      out << "]";
    }

    return out;
  }

  // ==============================================

  template <char OP, typename L>
  std::ostream &operator<<(std::ostream &sout, expr_unary<OP, L> const &expr) {
    return sout << OP << expr.l;
  }

  template <char OP, typename L, typename R>
  std::ostream &operator<<(std::ostream &sout, expr<OP, L, R> const &expr) {
    return sout << "(" << expr.l << " " << OP << " " << expr.r << ")";
  }

  // ==============================================

  template <typename F, typename... A>
  std::ostream &operator<<(std::ostream &out, expr_call<F, A...> const &) {
    return out << "mapped"; //array<value_type, std::decay_t<A>::rank>(x);
  }

} // namespace nda
