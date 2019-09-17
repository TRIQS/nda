#include "./permutation.hpp"

#define FORCEINLINE __inline__ __attribute__((always_inline))

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
  // no ellipsis : q = n
  // Args          = long, long, ellipsis,            long, range
  //  q               0      1      2                   3     4
  // Expanded Args = long, long, range, range, range, long, range
  //  n               0      1      2     3      4      5     6
  // e_pos = 2, e_pos + e_len  = 5
  // n will be 0 1 2 3 4 5 6
  // q will be 0 1 2 2 2 3 4
  // p -> n : 2 3 4 5
  // p -> q : 2 2 2 4

  // ----------     Computation of the position of the ellipsis in the args ----------------------
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
    return impl::ellipsis_position<Args...>(std::index_sequence_for<Args...>{});
  }

  //  ------------ Relation  n -> q, given the position and length of the ellipsis -----------
  // e_pos : ellipsis position
  // e_len : ellipsis length
  // return q
  constexpr int q_of_n(int n, int e_pos, int e_len) {
    if (n <= e_pos) return n; // if no ellipsis, e_pos is 128 = infty
    if (n >= (e_pos + e_len))
      return n - (e_len - 1);
    else
      return e_pos;
  }

  // ------------- The map  p-> n -------------------------
  // args_is_range : for each q, True iif the args is a range, or range_all or ellipsis
  // e_pos : ellipsis position
  // e_len : ellipsis length
  // return the map p-> n as a std::array : result[p] = n
  // for each index p, what is its position in the original idx_map
  //
  template <int N, int P, size_t Q> // Q is a size_t for matching std::array
  constexpr std::array<int, P> n_of_p_map(std::array<bool, Q> const &args_is_range, int e_pos, int e_len) {
    auto result = nda::make_initialized_array<P>(0);
    for (int n = 0, c = 0; n < N; ++n) {
      int q = q_of_n(n, e_pos, e_len);
      if (args_is_range[q]) result[c++] = n;
    }
    return result;
  }

  // ------------- The map  p-> n -------------------------
  // same as before except that it returns p-> q instead of p-> n
  //
  template <int N, int P, size_t Q>
  constexpr std::array<int, P> q_of_p_map(std::array<bool, Q> const &args_is_range, int e_pos, int e_len) {
    auto result = nda::make_initialized_array<P>(0);
    for (int n = 0, c = 0; n < N; ++n) {
      int q = q_of_n(n, e_pos, e_len);
      if (args_is_range[q]) result[c++] = q;
    }
    return result;
  }

  // ------------- The map n->p -------------------------
  // n_of_p : the map n->p
  // return the (pseudo) inverse map
  // n -> p or -1 if n is the index of a long argument
  template <size_t N, size_t P>
  constexpr std::array<int, N> p_of_n_map(std::array<int, P> const &n_of_p) {
    auto result = nda::make_initialized_array<N>(-1);
    for (size_t p = 0; p < P; ++p) result[n_of_p[p]] = p;
    return result;
  }

  // --------------  Slice the stride_order -----------------------
  // stride_order : the permutation stride_order. stride_order[0] : slowest, etc...
  // n_of_p : the map p-> n
  // return : the new stride_order of the sliced map
  template <size_t P, size_t N>
  constexpr std::array<int, P> sliced_mem_stride_order(std::array<int, N> const &stride_order_in, std::array<int, P> const &n_of_p) {
    //if (stride_order_in == 0) return 0; // quick decision C-> C
    auto stride_order = nda::make_initialized_array<P>(0);
    auto p_of_n       = p_of_n_map<N>(n_of_p); // reverse the map
    for (size_t i = 0, ip = 0; i < N; ++i) {   // i : index of the n
      int n = stride_order_in[i];              // n traverses the N in the order of the stride_order. Slowest first.
      int p = p_of_n[n];                       // n->p or -1 is n is a long argument
      if (p != -1) stride_order[ip++] = p;     // if p is fine, it is the next
    }
    return stride_order;
  }

  // -------------- Slice the stride_order info flags-----------
  // args_is_range_all : for each q, True iif the args is a range_all or an ellipsis [NO range here !]
  // stride_order : the stride_order of idx_map to be slided
  // Nlast : position, in q, of the argument corresponding to the fastest stride
  // layout_prop : to be sliced
  //
  template <size_t Q, size_t N>
  constexpr layout_prop_e slice_layout_prop(bool has_only_rangeall_and_long, std::array<bool, Q> const &args_is_range_all, int Nlast,
                                            std::array<int, N> const &stride_order_in, layout_prop_e layout_prop) {

    if (not has_only_rangeall_and_long) return layout_prop_e::none;
    // count the number of times 1 0 appears args_in_range
    // we must traverse in the order of the stride_order ! (in memory order, slowest to fastest)
    int n_10_pattern = 0;
    for (size_t i = 1; i < Q; ++i) {
      int n = stride_order_in[i];
      if (args_is_range_all[n - 1] and (not args_is_range_all[n])) ++n_10_pattern;
    }
    bool rangeall_are_grouped_in_memory             = (n_10_pattern <= 1);
    bool rangeall_are_grouped_in_memory_and_fastest = (n_10_pattern == 0);
    bool last_is_rangeall                           = args_is_range_all[Nlast];

    layout_prop_e r = layout_prop_e::none;

    if ((layout_prop & layout_prop_e::contiguous) and rangeall_are_grouped_in_memory_and_fastest) r = r | layout_prop_e::contiguous;
    if ((layout_prop & layout_prop_e::strided_1d) and rangeall_are_grouped_in_memory) r = r | layout_prop_e::strided_1d;
    if ((layout_prop & layout_prop_e::smallest_stride_is_one) and last_is_rangeall) r = r | layout_prop_e::smallest_stride_is_one;

    return r;
  }

  // ------------  Small pieces of code for the fold in functions below, with dispatch on type --------------------------------------
  // offset
  // first arg : the n-th argument (after expansion of the ellipsis)
  // second arg : s_n  : stride[n] of the idx_map
  FORCEINLINE long get_offset(long R, long s_n) { return R * s_n; }
  FORCEINLINE long get_offset(range const &R, long s_n) { return R.first() * s_n; }
  FORCEINLINE long get_offset(range_all, long) { return 0; }

  // length. Same convention
  // second arg : l_n  : length[n] of the idx_map
  FORCEINLINE long get_l(range const &R, long l_n) {
    return ((R.last() == -1 ? l_n : R.last()) - R.first() + R.step() - 1) / R.step(); // python behaviour
  }
  FORCEINLINE long get_l(range_all, long l_n) { return l_n; }

  // strides
  FORCEINLINE long get_s(range const &R, long s_n) { return s_n * R.step(); }
  FORCEINLINE long get_s(range_all, long s_n) { return s_n; }

  // temporary debug
  template <int... R>
  struct debug {};
#define PRINT(...) debug<__VA_ARGS__>().zozo;

  // ----------------------------- slice of index map  : implementation function ----------------------------------------------

  // Ns, Ps, Qs : sequence indices for size N, P, Q
  // IdxMap : type of the indexmap idx
  // Arg : arguments of the slice
  // returns : a pair:  (offset, new sliced stride_order)
  //
  template <size_t... Ns, size_t... Ps, size_t... Qs, typename IdxMap, typename... Args>
  FORCEINLINE auto slice_stride_order_impl(std::index_sequence<Ps...>, std::index_sequence<Ns...>, std::index_sequence<Qs...>, IdxMap const &idxm,
                                           Args const &... args) {

#ifdef NDA_ENFORCE_BOUNDCHECK
    details::assert_in_bounds(idxm.rank(), idxm.lengths().data(), args...);
#endif

    static_assert(IdxMap::rank() == sizeof...(Ns), "Internal error");
    static_assert(sizeof...(Args) == sizeof...(Qs), "Internal error");

    static constexpr int N     = sizeof...(Ns);
    static constexpr int P     = sizeof...(Ps);
    static constexpr int Q     = sizeof...(Qs);
    static constexpr int e_len = N - Q + 1; // len of ellipsis : how many ranges are missing
    static constexpr int e_pos = ellipsis_position<Args...>();

    // Pattern of the arguments. 1 for a range/range_all/ellipsis, 0 for long
    static constexpr std::array<bool, Q> args_is_range{(std::is_same_v<Args, range> or std::is_base_of_v<range_all, Args>)...};
    static constexpr std::array<bool, Q> args_is_range_all{(std::is_base_of_v<range_all, Args>)...};

    static constexpr std::array<int, P> n_of_p = n_of_p_map<N, P>(args_is_range, e_pos, e_len);
    static constexpr std::array<int, P> q_of_p = q_of_p_map<N, P>(args_is_range, e_pos, e_len);

    static_assert(n_of_p.size() == P, "Oops"); // sanity check
    static_assert(q_of_p.size() == P, "Oops");

    auto argstie = std::tie(args...);

    //	PRINT(bitsP, Qn_of_p_map(bitsP, 0, e_pos, e_len)); PRINT(e_pos); PRINT(e_len);

    std::array<long, P> len{get_l(std::get<q_of_p[Ps]>(argstie), std::get<n_of_p[Ps]>(idxm.lengths()))...};
    std::array<long, P> str{get_s(std::get<q_of_p[Ps]>(argstie), std::get<n_of_p[Ps]>(idxm.strides()))...};

    static constexpr std::array<int, P> mem_stride_order = sliced_mem_stride_order(IdxMap::stride_order, n_of_p);

    // Compute the new layout_prop
    static constexpr bool has_only_rangeall_and_long = ((std::is_constructible_v<long, Args> or std::is_base_of_v<range_all, Args>)and...);

    static constexpr layout_prop_e li = slice_layout_prop(
       has_only_rangeall_and_long, args_is_range_all, q_of_n(IdxMap::stride_order[N - 1], e_pos, e_len), IdxMap::stride_order, IdxMap::layout_prop);

    long offset = (get_offset(std::get<q_of_n(Ns, e_pos, e_len)>(argstie), std::get<Ns>(idxm.strides())) + ... + 0);

    return std::make_pair(offset, idx_map<P, permutations::encode(mem_stride_order), li>{len, str});
    //return idx_map<P, permutations::encode(stride_order)>{len, str, offset};
  }

  // ----------------------------- slice of index map ----------------------------------------------
  //
  template <typename IdxMap, typename... T>
  FORCEINLINE decltype(auto) slice_stride_order(IdxMap const &idxm, T const &... x) {

    static constexpr int n_args_ellipsis = ((std::is_same_v<T, ellipsis>)+...);
    static constexpr int n_args_long     = (std::is_constructible_v<long, T> + ...);

    static_assert(n_args_ellipsis <= 1, "Only one ellipsis argument is authorized");
    static_assert((sizeof...(T) <= IdxMap::rank()), "Incorrect number of arguments in array call ");
    static_assert((n_args_ellipsis == 1) or (sizeof...(T) == IdxMap::rank()), "Incorrect number of arguments in array call ");

    return slice_stride_order_impl(std::make_index_sequence<IdxMap::rank() - n_args_long>{}, std::make_index_sequence<IdxMap::rank()>{},
                                   std::make_index_sequence<sizeof...(T)>{}, idxm, x...);
  }

  // ----------------------------- slice of index map ----------------------------------------------

  //   Ns, Ps, Qs : sequence indices for size N, P, Q
  // IdxMap : type of the indexmap idx
  // Arg : arguments of the slice
  // returns : a new sliced idx_map, with computed rank, stride_order
  //
  //template <size_t... Ns, typename IdxMap, typename... Args>
  //FORCEINLINE auto offset_of_slice(std::index_sequence<Ps...>, IdxMap const &idxm, Args const &... args) {

  //static_assert(IdxMap::rank() == sizeof...(Ns), "Internal error");
  //static constexpr int N     = sizeof...(Ns);
  //static constexpr int Q     = sizeof...(Args);
  //static constexpr int e_len = N - Q + 1; // len of ellipsis : how many ranges are missing
  //static constexpr int e_pos = ellipsis_position<Args...>();

  //auto argstie = std::tie(args...);
  //long offset  = idxm.offset() + (get_offset(std::get<q_of_n(Ns, e_pos, e_len)>(argstie), std::get<Ns>(idxm.strides())) + ... + 0);
  //return offset;
  //}

} // namespace nda::slice_static
