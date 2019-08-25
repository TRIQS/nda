#include "./range.hpp"

#define FORCEINLINE __inline__ __attribute__((always_inline))

namespace nda {
  template <int Rank> class idx_map;
};

namespace nda::slice_static {

  // Notations
  // RankIn : rank of the original idx_map, hence len of its len and stri.
  // RankOut : rank of the resulting idx_map
  // NArgs : number of arguments of the slices.
  //         RankIn - NArgs is the length of the ellipsis, 0 if no ellipsis in the argument.
  // N : 0, ..., RankIn : index the original idx_map
  // P : 0, ..., RankOut : index the resulting idx_map
  // L : 0, ..., RankIn - RankOut : index the long argument
  // Q : 0, ..., NArgs : index the arguments
  //
  // Arguments are of the form e.g. for RankIn = 6 : long, long , Ellipsis, long , Range
  // The bit corresponding bit pattern (argsbits) is 00101 (in fact stored in reverse)
  // They correspond to a call (with Ellipsis expanded)  : long, long, RangeAll, RangeAll, long, Range
  //
  // P are the indices of the non long arguments (after ellipsis expansion)
  // We have compile time maps :
  // P -> N   and P-> Q
  // L -> N   and L-> Q

  // example :
  // no ellipsis : q = n
  // Args          = long, long, ellipsis,            long, range
  //  q               0      1      2                   3     4
  // Expanded Args = long, long, range, range, range, long, range
  //  n 	            0      1      2     3      4      5     6
  // ellipsis_position = 2, ellipsis_position + ellipsis_length  = 5
  // n will be 0 1 2 3 4 5 6
  // q will be 0 1 2 2 2 3 4
  // argbits 00101
  // P -> q,n : 0 -> 2,2, 1 -> 2,3, 2-> 2,4, 3 -> 4,6
  // since P < 4

  // Impl detail.
  template <typename... Args, size_t... Is> constexpr int impl_ellipsis_position(std::index_sequence<Is...>) {
    // We know that there is at most one ellipsis.
    int r = ((std::is_same_v<Args, ellipsis> ? int(Is) + 1 : 0) + ...); // position + 1 or 0
    return (r == 0 ? 128 : r - 1);
  }

  // position of the ellipsis in the argument list if there is an ellipsis or 128 if not
  template <typename... Args> constexpr int compute_ellipsis_position() {
    return impl_ellipsis_position<Args...>(std::index_sequence_for<Args...>{});
  }

  // Q given N
  constexpr int Q_of_N(int n, int ellipsis_position, int ellipsis_length) {
    if (n <= ellipsis_position) return n; // if no ellipsis, ellipsis_position is 128 = infty
    if (n >= (ellipsis_position + ellipsis_length))
      return n - (ellipsis_length - 1);
    else
      return ellipsis_position;
  }

  // map P or L -> N and Q
  // The map depends on the position and len of the ellipsis
  // N, Q < 64 : returns them in a single int (< 64*64).
  // NB : if args_as_bits is inverted ( reverse 1 <-> 0), it finds the maps L -> N Q instead of P-> N Q
  constexpr int QN_of_P(uint64_t args_as_bits, int P, int ellipsis_position, int ellipsis_length) {
    // we need to find the P th non long argument, but also correct for the ellipsis
    // so q iterate on the arguments, n on the dimensions of the original idx_map
    for (int n = 0, c = -1; n < 64; ++n) {
      int q = Q_of_N(n, ellipsis_position, ellipsis_length);
      if (args_as_bits & (1ull << q)) ++c; // c count the number of non long we encounter
      if (c == P) return n + 64 * q;
    }          // it must stop by construction because P < rank of the output idx_map.
    return -1; // remove warning
  }

  // map P -> N
  constexpr int N_of_P(uint64_t args_as_bits, int P, int ellipsis_position, int ellipsis_length) {
    return QN_of_P(args_as_bits, P, ellipsis_position, ellipsis_length) % 64;
  }
  // map P -> Q
  constexpr int Q_of_P(uint64_t args_as_bits, int P, int ellipsis_position, int ellipsis_length) {
    return QN_of_P(args_as_bits, P, ellipsis_position, ellipsis_length) / 64;
  }

  // Small pieces of code for the fold in functions below, with dispatch on type.
  // offset
  FORCEINLINE long get_offset(long R, long s_N) { return R * s_N; }
  FORCEINLINE long get_offset(range const &R, long s_N) { return R.first() * s_N; }
  FORCEINLINE long get_offset(range_all, long) { return 0; }

  // length
  FORCEINLINE long get_l(range const &R, long l_N) {
    return ((R.last() == -1 ? l_N : R.last()) - R.first() + R.step() - 1) / R.step(); // python behaviour
  }
  FORCEINLINE long get_l(range_all, long l_N) { return l_N; }
  long get_l(long, long) = delete; // can not happen

  // strides
  FORCEINLINE long get_s(range const &R, long s_N) { return s_N * R.step(); }
  FORCEINLINE long get_s(range_all, long s_N) { return s_N; }
  long get_s(long, long) = delete; // can not happen

  template <int... R> struct debug {};
  #define PRINT(...) debug<__VA_ARGS__>().zozo;

  // main function: slice the idx_map
  template <int R, size_t... N, size_t... P, size_t... Q, typename... Args>
  FORCEINLINE auto slice(std::index_sequence<P...>, std::index_sequence<N...>, std::index_sequence<Q...>, idx_map<R> const &idx,
                         Args const &... args) {

    static constexpr int e_len          = R - sizeof...(Args) + 1; // len of ellipsis : how many ranges are missing
    static constexpr int e_pos          = compute_ellipsis_position<Args...>();
    static constexpr int rank_of_result = sizeof...(P);

    // compute the bit pattern for the argument, cf above
    constexpr uint64_t bitsP = (((std::is_same_v<Args, range> or std::is_base_of_v<range_all, Args >) ? (1 << Q) : 0) + ...);

    auto argstie = std::tie(args...);

    //	PRINT(bitsP, QN_of_P(bitsP, 0, e_pos, e_len)); PRINT(e_pos); PRINT(e_len);

    long offset = idx.offset() + (get_offset(std::get<Q_of_N(N, e_pos, e_len)>(argstie), std::get<N>(idx.strides())) + ... + 0);

    std::array<long, rank_of_result> len{
       get_l(std::get<Q_of_P(bitsP, P, e_pos, e_len)>(argstie), std::get<N_of_P(bitsP, P, e_pos, e_len)>(idx.lengths()))...};

    std::array<long, rank_of_result> str{
       get_s(std::get<Q_of_P(bitsP, P, e_pos, e_len)>(argstie), std::get<N_of_P(bitsP, P, e_pos, e_len)>(idx.strides()))...};

    return idx_map<rank_of_result>{len, str, offset};
  }

} // namespace nda::slice_static
