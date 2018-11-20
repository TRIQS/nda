#include "./range.hpp"

#define FORCEINLINE __inline__ __attribute__((always_inline))

namespace nda::details {

  // -------------------- ellipsis_position ---------------------

  //constexpr function to compute the position of the ellipsis in a list of argument.
  // if there is an ellipsis in the args, otherwise, it returns -1
  template <typename... Args, size_t... Is> constexpr int ellipsis_position_impl(std::index_sequence<Is...>) {
    return ((std::is_same_v<Args, ellipsis> ? int(Is) + 1 : 0) + ...) - 1;
  }

  template <typename... Args> constexpr int ellipsis_position() { return ellipsis_position_impl<Args...>(std::index_sequence_for<Args...>{}); }

  // -------------------- slice_worker ---------------------
  //
  // A worker to compute the slice.
  // It takes the length, slice of the idx_map to be sliced
  struct slice_worker {
    long const *li, *si; // length and slice of input slice
    long *lo, *so;       // length and slice of the result
    int *imap;           //

    // result of the worker
    long &offset;     // resulting offset
    int N = 0, P = 0; // internal variable
    uint32_t error_code = 0;

    void _check_BC(long N, long ind, long B, long ind_min = 0) {
#ifdef TRIQS_ARRAYS_ENFORCE_BOUNDCHECK
      if (!((ind >= ind_min) && (ind < B))) error_code += (1ul << N);
#endif
    }

    private:
    FORCEINLINE slice_worker &operator<<(long R) {
      _check_BC(N, R, li[N]);
      offset += R * si[N];
      //std::cout << "off set  :  " << offset << " " << N << " R : " << R << " si " << si[N] << std::endl;
      imap[N] = -1;
      ++N;
      return *this;
    }

    FORCEINLINE slice_worker &operator<<(range R) {
      _check_BC(N, R.first(), li[N]);
      lo[P] = ((R.last() == -1 ? li[N] : R.last()) - R.first() + R.step() - 1) / R.step(); // python behaviour
      _check_BC(N, R.first() + (lo[P] != 0 ? (lo[P] - 1) : 0) * R.step(), li[N], -1);
      so[P] = si[N] * R.step();
      offset += R.first() * si[N];
      //std::cout << "off set  :  " << offset << " " << N << " R : " << R << " si " << si[N] << std::endl;
      imap[N] = P;
      ++N, ++P;
      return *this;
    }

    // ellipsis is a total range, we can simplify the computation in that case...
    // FIXME ellipsis as an empyt struct, with a constexpr bool is_range_or_ellispsis (long + ellipsis) + a range_all
    // nda::vars::_ nda::vars:___ 1 or 3
    FORCEINLINE slice_worker &operator<<(range_all) {
      lo[P] = li[N];
      _check_BC(N, (lo[P] != 0 ? (lo[P] - 1) : 0), li[N], -1);
      so[P]   = si[N];
      imap[N] = P;
      ++N, ++P;
      return *this;
    }

    // detail function use in the implementation  of process below
    static constexpr size_t idx_map_compute_index(size_t Is, int EllipsisPosition, int EllipsisLength) {
      if (Is <= EllipsisPosition) return Is;
      if (Is >= EllipsisPosition + EllipsisLength) return Is - EllipsisLength + 1;
      return EllipsisPosition;
    }

    // implementation of process_static  
    template <typename... Args, size_t... Is> FORCEINLINE void process_impl(std::index_sequence<Is...>, Args const &... args) {

      static constexpr int ellipsis_loss = sizeof...(Is) - sizeof...(Args); // len of ellipsis : how many ranges are missing

      if constexpr (ellipsis_loss == 0) {
        ((*this) << ... << args);
      } else {
        auto tuargs = std::tie(args...);
        ((*this) << ... << std::get<idx_map_compute_index(Is, ellipsis_position<Args...>(), ellipsis_loss + 1)>(tuargs));
      }
    }

    // implementation of process_dyn
    template <size_t IStart, typename TuArgs, size_t... Is> FORCEINLINE void process_partial_impl(std::index_sequence<Is...>, TuArgs &&tuargs) {
      ((*this) << ... << std::get<Is + IStart>(tuargs));
    }

    public:

    // --------------  process_dyn --------------------

    template <typename... Args> FORCEINLINE void process_dynamic(int rank, Args const &... args) {
      static constexpr int ep = ellipsis_position<Args...>();
      auto tuargs             = std::tie(args...);
      
      // process all arguments before the ellipsis, including the ellipsis once.
      process_partial_impl<0>(std::make_index_sequence<ep+1>{}, tuargs);
     
      // process the ellipsis as much as needed to fill the missing term 
      int ellipsis_loss = rank - sizeof...(Args); // len of ellipsis : how many ranges are missing
      for (int i=0; i< ellipsis_loss; ++i) (*this) << ellipsis{};
      
      // process all the arguments after the ellipsis
      process_partial_impl<ep+1>(std::make_index_sequence<sizeof...(Args) - ep>{}, tuargs);
    }

    // --------------  process_dynamic --------------------

    template <int Rank, typename... Args> FORCEINLINE void process_static(Args const &... args) {
      process_impl(std::make_index_sequence<Rank>{}, args...);
    }
  };

} // namespace nda::details
