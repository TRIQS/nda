#include "./range.hpp"
#include "./permutation.hpp"

#define FORCEINLINE __inline__ __attribute__((always_inline))

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
  // e_pos = 2, e_pos + e_len  = 5
  // n will be 0 1 2 3 4 5 6
  // q will be 0 1 2 2 2 3 4
  // argbits 00101
  // P -> q,n : 0 -> 2,2, 1 -> 2,3, 2-> 2,4, 3 -> 4,6
  // since P < 4

  // Impl detail.
  template <typename... Args, size_t... Is> constexpr int impl_e_pos(std::index_sequence<Is...>) {
    // We know that there is at most one ellipsis.
    int r = ((std::is_same_v<Args, ellipsis> ? int(Is) + 1 : 0) + ...); // position + 1 or 0
    return (r == 0 ? 128 : r - 1);
  }

  // position of the ellipsis in the argument list if there is an ellipsis or 128 if not
  template <typename... Args> constexpr int compute_e_pos() { return impl_e_pos<Args...>(std::index_sequence_for<Args...>{}); }

  // Q given N
  constexpr int Q_of_N(int n, int e_pos, int e_len) {
    if (n <= e_pos) return n; // if no ellipsis, e_pos is 128 = infty
    if (n >= (e_pos + e_len))
      return n - (e_len - 1);
    else
      return e_pos;
  }

  // map P -> N
  template <int RankN, int RankP, size_t RankQ>
  constexpr std::array<int, RankP> N_of_P(std::array<bool, RankQ> const &args_as_bits, int e_pos, int e_len) {
    auto result = nda::make_initialized_array<RankP>(0);
    // we need to find the P th non long argument, but also correct for the ellipsis
    // so q iterate on the arguments, n on the dimensions of the original idx_map
    for (int n = 0, c = 0; n < RankN; ++n) {
      int q = Q_of_N(n, e_pos, e_len);
      if (args_as_bits[q]) result[c++] = n;
    }
    return result;
  }

  // map P -> Q : almost the same as before except I store Q
  template <int RankN, int RankP, size_t RankQ>
  constexpr std::array<int, RankP> Q_of_P(std::array<bool, RankQ> const &args_as_bits, int e_pos, int e_len) {
    auto result = nda::make_initialized_array<RankP>(0);
    for (int n = 0, c = 0; n < RankN; ++n) {
      int q = Q_of_N(n, e_pos, e_len);
      if (args_as_bits[q]) result[c++] = q;
    }
    return result;
  }

  // map P-> N given N->P : use 0xF as the invalid
  template <int RankN, int RankP> constexpr std::array<int, RankN> P_of_N(std::array<int, RankP> const &n_of_p) {
    auto result = nda::make_initialized_array<RankN>(-1);
    for (int p = 0; p < RankP; ++p) {
      result[n_of_p[p]] = p;
      return result;
    }
  }

  template <int RankP, int RankN>
  constexpr std::array<int, RankP> sliced_layout(std::array<int, RankN> const &layout, std::array<int, RankP> const &n_of_p) {
    auto result = nda::make_initialized_array<RankP>(0);
    auto p_of_n = P_of_N<RankN>(n_of_p);
    for (int i = 0, ip = 0; i < RankP; ++i) {
      int n = layout[n]; // n traverses the N in the order of the layout. Slowest first.
      int p = p_of_n[n]; // N -> P
      if (p != -1) result[ip] = p;
    }
    return result;
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
  template <typename IdxMap, size_t... N, size_t... P, size_t... Q, typename... Args>
  FORCEINLINE auto slice(std::index_sequence<P...>, std::index_sequence<N...>, std::index_sequence<Q...>, IdxMap const &idx, Args const &... args) {
    static_assert(IdxMap::rank() == sizeof...(N), "Internal error");

    static constexpr int Rank           = IdxMap::rank();
    static constexpr int e_len          = Rank - sizeof...(Args) + 1; // len of ellipsis : how many ranges are missing
    static constexpr int e_pos          = compute_e_pos<Args...>();
    static constexpr int rank_of_result = sizeof...(P);

    // Pattern of the arguments. 1 for a range/range_all/ellipsis, 0 for long
    static constexpr std::array<bool, sizeof...(Q)> args_as_bits{(std::is_same_v<Args, range> or std::is_base_of_v<range_all, Args>)...};

    static constexpr std::array<int, rank_of_result> n_of_p = N_of_P<Rank, rank_of_result>(args_as_bits, e_pos, e_len);
    static constexpr std::array<int, rank_of_result> q_of_p = Q_of_P<Rank, rank_of_result>(args_as_bits, e_pos, e_len);

    static_assert(n_of_p.size() == rank_of_result, "Oops"); // sanity check
    static_assert(q_of_p.size() == rank_of_result, "Oops");

    auto argstie = std::tie(args...);

    //	PRINT(bitsP, QN_of_P(bitsP, 0, e_pos, e_len)); PRINT(e_pos); PRINT(e_len);

    long offset = idx.offset() + (get_offset(std::get<Q_of_N(N, e_pos, e_len)>(argstie), std::get<N>(idx.strides())) + ... + 0);

    std::array<long, rank_of_result> len{get_l(std::get<q_of_p[P]>(argstie), std::get<n_of_p[P]>(idx.lengths()))...};
    std::array<long, rank_of_result> str{get_s(std::get<q_of_p[P]>(argstie), std::get<n_of_p[P]>(idx.strides()))...};

    // FIXME
    //return idx_map<rank_of_result, sliced_layout(Layout, bitsP, rank_of_result, e_pos, e_len), slided_flags(Flags)>{len, str, offset};
    return idx_map<rank_of_result, 0, 0>{len, str, offset};
  }

} // namespace nda::slice_static
