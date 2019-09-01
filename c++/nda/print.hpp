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
  template <int Rank, uint64_t Layout, uint64_t Flags>
  std::ostream &operator<<(std::ostream &out, idx_map<Rank, Layout, Flags> const &x) {
    return out << "  Lengths  : " << x.lengths() << "\n"
               << "  Strides  : " << x.strides() << "\n"
               << "  Offset   : " << x.offset() << "\n"
               << "  Layout   : " << x.layout << "\n"
               << "  Flags   :  " << (flags::has_contiguous(x.flags) ? "contiguous   " : " ") << (flags::has_strided(x.flags) ? "strided   " : " ")
               << (flags::has_smallest_stride_is_one(x.flags) ? "smallest_stride_is_one   " : " ")
               << (flags::has_zero_offset(x.flags) ? "zero_offset   " : " ") << "\n";
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
      for (size_t j = 0; j < a.size(); ++j) out << a.data_start[j];
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
  std::ostream &operator<<(std::ostream &out, expr_call<F, A...> const &x) {
    return out << "mapped"; //array<value_type, std::decay_t<A>::rank>(x);
  }

} // namespace nda
