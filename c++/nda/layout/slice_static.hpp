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

#include "./permutation.hpp"

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

  // ------------- The map  p-> n -------------------------
  // args_is_range : for each q, True iif the args is a range, or range_all or ellipsis
  // e_pos : ellipsis position
  // e_len : ellipsis length
  // return the map p-> n as a std::array : result[p] = n
  // for each index p, what is its position in the original idx_map
  //
  template <int N, int P, size_t Q> // Q is a size_t for matching std::array
  constexpr std::array<int, P> n_of_p_map(std::array<bool, Q> const &args_is_range, int e_pos, int e_len) {
    auto result = stdutil::make_initialized_array<P>(0);
    int p       = 0;
    for (int n = 0; n < N; ++n) {
      int q = q_of_n(n, e_pos, e_len);
      if (args_is_range[q]) result[p++] = n;
    }
    if (p != P) throw std::logic_error("Internal Error");
    return result;
  }

  // ------------- The map  p-> q -------------------------
  // same as before except that it returns p-> q instead of p-> n
  //
  template <int N, int P, size_t Q>
  constexpr std::array<int, P> q_of_p_map(std::array<bool, Q> const &args_is_range, int e_pos, int e_len) {
    auto result = stdutil::make_initialized_array<P>(0);
    int p       = 0;
    for (int n = 0; n < N; ++n) {
      int q = q_of_n(n, e_pos, e_len);
      if (args_is_range[q]) result[p++] = q;
    }
    if (p != P) throw std::logic_error("Internal Error");
    return result;
  }

  // ------------- The map n->p -------------------------
  // n_of_p : the map n->p
  // return the (pseudo) inverse map
  // n -> p or -1 if n is the index of a long argument
  template <size_t N, size_t P>
  constexpr std::array<int, N> p_of_n_map(std::array<int, P> const &n_of_p) {
    auto result = stdutil::make_initialized_array<N>(-1);
    for (size_t p = 0; p < P; ++p) result[n_of_p[p]] = p;
    return result;
  }

  // --------------  Slice the stride_order -----------------------
  // stride_order : the permutation stride_order. stride_order[0] : slowest, etc...
  // n_of_p : the map p-> n
  // return : the new stride_order of the sliced map
  template <size_t P, size_t N>
  constexpr std::array<int, P> sliced_mem_stride_order(std::array<int, N> const &stride_order_in, std::array<int, P> const &n_of_p) {
    // quick short cut : does it really impact compilation time ?
    //if (stride_order_in == 0) return 0; // quick decision C-> C
    auto stride_order = stdutil::make_initialized_array<P>(0);
    auto p_of_n       = p_of_n_map<N>(n_of_p); // reverse the map
    // traverse the n in the order given by stride_order_in and keep them if they are associated to a p
    for (int i = 0, ip = 0; i < N; ++i) {  // i : index of the n
      int n = stride_order_in[i];          // n traverses the N in the order of the stride_order. Slowest first.
      int p = p_of_n[n];                   // n->p or -1 is n is a long argument
      if (p != -1) stride_order[ip++] = p; // if p is fine, it is the next
    }
    return stride_order;
  }

  // -------------- Slice the stride_order info flags-----------
  // P : as before
  // has_only_rangeall_and_long : as the name says...
  // args_is_range_all : for each q, True iif the args is a range_all or an ellipsis [NO range here !]
  // stride_order : the stride_order of idx_map to be slided
  // Nlast : position, in q, of the argument corresponding to the fastest stride
  // layout_prop : to be sliced
  //
  template <size_t Q, size_t N>
  constexpr layout_prop_e slice_layout_prop(int P, bool has_only_rangeall_and_long, std::array<bool, Q> const &args_is_range_all,
                                            std::array<int, N> const &stride_order, layout_prop_e layout_prop, int e_pos, int e_len) {

    // if we have some ranges, we give up
    if (not has_only_rangeall_and_long) {
      if (P == 1) // rank one is special. Alyways at least 1d strided
        return layout_prop_e::strided_1d;
      else
        return layout_prop_e::none;
    }
    // count the number of times 1 0 appears args_in_range
    // we must traverse in the order of the stride_order ! (in memory order, slowest to fastest)
    int n_1_blocks                = 0;
    bool previous_arg_is_rangeall = false;
    for (int i_n = 0; i_n < int(N); ++i_n) {
      int q                = q_of_n(stride_order[i_n], e_pos, e_len);
      bool arg_is_rangeall = args_is_range_all[q];
      if (arg_is_rangeall and (not previous_arg_is_rangeall)) ++n_1_blocks;
      previous_arg_is_rangeall = arg_is_rangeall;
    }
    // in mem order. e.g. (long, all, all, long) or (all, all, long), but not (all, long, all)
    bool rangeall_are_grouped_in_memory = (n_1_blocks <= 1);
    bool last_is_rangeall               = previous_arg_is_rangeall;

    if (has_contiguous(layout_prop) and rangeall_are_grouped_in_memory and last_is_rangeall) return layout_prop_e::contiguous;
    if (has_strided_1d(layout_prop) and rangeall_are_grouped_in_memory) return layout_prop_e::strided_1d;
    if (has_smallest_stride_is_one(layout_prop) and last_is_rangeall) return layout_prop_e::smallest_stride_is_one;

    return layout_prop_e::none;
  }

  // ------------  Small pieces of code for the fold in functions below, with dispatch on type --------------------------------------
  // offset
  // first arg : the n-th argument (after expansion of the ellipsis)
  // second arg : s_n  : stride[n] of the idx_map
  FORCEINLINE long get_offset(long R, long s_n) { return R * s_n; }
  FORCEINLINE long get_offset(range const &R, long s_n) { return R.first() * s_n; }
  FORCEINLINE long get_offset(range::all_t, long) { return 0; }

  // length. Same convention
  // second arg : l_n  : length[n] of the idx_map
  FORCEINLINE long get_l(range const &R, long l_n) {
    auto last = (R.last() == -1 and R.step() > 0) ? l_n : R.last();
    return range(R.first(), last, R.step()).size();
  }
  FORCEINLINE long get_l(range::all_t, long l_n) { return l_n; }

  // strides
  FORCEINLINE long get_s(range const &R, long s_n) { return s_n * R.step(); }
  FORCEINLINE long get_s(range::all_t, long s_n) { return s_n; }

  // compile time print debug
  //template <int... R>
  //struct debug {};
  //#define PRINT(...) debug<__VA_ARGS__>().zozo;

  // ----------------------------- slice of index map  : implementation function ----------------------------------------------

  // Ns, Ps, Qs : sequence indices for size N, P, Q
  // IdxMap : type of the indexmap idx
  // Arg : arguments of the slice
  // returns : a pair:  (offset, new sliced stride_order)
  //
  // FIXME : Q only needed, not Qs
  template <size_t... Ps, size_t... Ns, size_t... Qs, typename IdxMap, typename... Args>
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
    static constexpr std::array<bool, Q> args_is_range{(std::is_same_v<Args, range> or std::is_base_of_v<range::all_t, Args>)...};
    static constexpr std::array<bool, Q> args_is_range_all{(std::is_base_of_v<range::all_t, Args>)...};

    static constexpr std::array<int, P> n_of_p = n_of_p_map<N, P>(args_is_range, e_pos, e_len);
    static constexpr std::array<int, P> q_of_p = q_of_p_map<N, P>(args_is_range, e_pos, e_len);

    static_assert(n_of_p.size() == P, "Oops"); // sanity check
    static_assert(q_of_p.size() == P, "Oops");

    auto argstie = std::tie(args...);

    //	PRINT(bitsP, Qn_of_p_map(bitsP, 0, e_pos, e_len)); PRINT(e_pos); PRINT(e_len);

    // Compute the offset of the pointer
    long offset = (get_offset(std::get<q_of_n(Ns, e_pos, e_len)>(argstie), std::get<Ns>(idxm.strides())) + ... + 0);

    // Compute the new len and strides
    std::array<long, P> len{get_l(std::get<q_of_p[Ps]>(argstie), std::get<n_of_p[Ps]>(idxm.lengths()))...};
    std::array<long, P> str{get_s(std::get<q_of_p[Ps]>(argstie), std::get<n_of_p[Ps]>(idxm.strides()))...};

    // Compute the new static_extents
    static constexpr std::array<int, P> new_static_extents{(args_is_range_all[q_of_p[Ps]] ? IdxMap::static_extents[n_of_p[Ps]] : 0)...};

    // The new Stride Order
    static constexpr std::array<int, P> mem_stride_order = sliced_mem_stride_order(IdxMap::stride_order, n_of_p);

    // Compute the new layout_prop
    static constexpr bool has_only_rangeall_and_long = ((std::is_constructible_v<long, Args> or std::is_base_of_v<range::all_t, Args>)and...);

    static constexpr layout_prop_e li =
       slice_layout_prop(P, has_only_rangeall_and_long, args_is_range_all, IdxMap::stride_order, IdxMap::layout_prop, e_pos, e_len);

    static constexpr uint64_t new_static_extents_encoded = encode(new_static_extents);
    static constexpr uint64_t mem_stride_order_encoded   = encode(mem_stride_order);
    return std::make_pair(offset, idx_map<P, new_static_extents_encoded, mem_stride_order_encoded, li>{len, str});
  }

  // ----------------------------- slice of index map ----------------------------------------------

  template <int R, uint64_t SE, uint64_t SO, layout_prop_e LP, typename... T>
  FORCEINLINE decltype(auto) slice_stride_order(idx_map<R, SE, SO, LP> const &idxm, T const &... x) {

    static constexpr int n_args_ellipsis = ((std::is_same_v<T, ellipsis>)+...);
    static constexpr int n_args_long     = (std::is_constructible_v<long, T> + ...); // any T I can construct a long from

    static_assert(n_args_ellipsis <= 1, "At most one ellipsis argument is authorized");
    static_assert((sizeof...(T) <= R + 1), "Incorrect number of arguments in array call ");
    static_assert((n_args_ellipsis == 1) or (sizeof...(T) == R), "Incorrect number of arguments in array call ");

    return slice_stride_order_impl(std::make_index_sequence<R - n_args_long>{}, std::make_index_sequence<R>{},
                                   std::make_index_sequence<sizeof...(T)>{}, idxm, x...);
  }

} // namespace nda::slice_static
