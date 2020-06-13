#pragma once

namespace nda::slice_static {

  // Notations for this file
  //
  // N : rank of the original idx_map
  // P : rank of the resulting idx_map
  // Q : number of arguments of the slice.
  //
  // n : 0, ..., N : index the dimensions of the original idx_map
  // p : 0, ..., P : index the dimensions of the resulting idx_map
  // q : 0, ..., Q : index the arguments
  //
  // NB : N - Q is the length of the ellipsis, 0 if no ellipsis in the argument.
  //
  // Arguments are of the form e.g. for N = 6 :
  //    long, long , Ellipsis, long , Range
  //  with Ellipsis expanded :
  //   long, long, RangeAll, RangeAll, long, Range
  //
  // p are the indices of the non long arguments (after ellipsis expansion)

  // We compute compile time map :  p -> n,  p -> q
  // example :
  // Args          = long, long, ellipsis,            long, range
  //  q               0      1      2                   3     4
  // Expanded Args = long, long, range, range, range, long, range
  //  n               0      1      2     3      4      5     6
  // e_pos = 2, e_pos + e_len  = 5
  // n will be 0 1 2 3 4 5 6
  // q will be 0 1 2 2 2 3 4
  // p -> n : 2 3 4 6
  // p -> q : 2 2 2 4

  // Case of ellipsis of zero size
  // Args          = long, long, range,  ellipsis,  long, range
  //  q               0      1      2       3         4     5
  // Expanded Args = long, long, range, long, range
  //  n               0      1      2     3      4
  // e_pos = 2, e_len  = 0
  // n will be 0 1 2 3 4
  // q will be 0 1 2 4 5
  // p -> n : 2 4
  // p -> q : 2 5

  // ----------     Computation of the position of the ellipsis in the args ----------------------
  // FIXME C++20
  namespace impl {
    template <typename... Args, size_t... Is>
    constexpr int ellipsis_position(std::index_sequence<Is...>) {
      // We know that there is at most one ellipsis.
      int r = ((std::is_same_v<Args, ellipsis> ? int(Is) + 1 : 0) + ...); // position + 1 or 0
      return (r == 0 ? 128 : r - 1);
    }
  } // namespace impl

  // position of the ellipsis in the argument list if there is an ellipsis or 128 if not
  template <typename... Args>
  constexpr int ellipsis_position() {
    return impl::ellipsis_position<Args...>(std::make_index_sequence<sizeof...(Args)>{}); //std::index_sequence_for<Args...>{});
  }

  //  ------------ Relation  n -> q, given the position and length of the ellipsis -----------
  // e_pos : ellipsis position
  // e_len : ellipsis length
  // return q
  //
  constexpr int q_of_n(int n, int e_pos, int e_len) {
    if (n < e_pos) return n; // if no ellipsis, e_pos is 128 = infty
    if (n < (e_pos + e_len)) // in the ellipsis
      return e_pos;
    else
      return n - (e_len - 1); // ok if e_len ==0  : n+1
  }


} // namespace nda::slice_static
