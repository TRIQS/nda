#pragma once

namespace nda {

  // Some basic functions depending only on the concept

  // --------------------------- make_regular ------------------------
  // FIXME MOVE THIS : A function, not a traits
  // general make_regular
  template <typename A>
  typename A::regular_t make_regular(A &&x) REQUIRES(is_ndarray_v<A>) {
    return std::forward<A>(x);
  }
  //template <typename A> regular_t<A> make_regular(A &&x) REQUIRES(is_ndarray_v<A>) { return std::forward<A>(x); }

} // namespace nda
