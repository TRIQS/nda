#pragma once

namespace nda {

  // Some basic functions depending only on the concept

  // --------------------------- make_regular ------------------------
  // FIXME MOVE THIS : A function, not a traits
  // general make_regular
  template <typename A>
  typename std::decay_t<A>::regular_t make_regular(A &&x) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
    return std::forward<A>(x);
  }
  //template <typename A> regular_t<A> make_regular(A &&x) REQUIRES(is_ndarray_v<A>) { return std::forward<A>(x); }

  // --------------------------- resize_or_check_if_view------------------------

  /** 
   * Resize if A is a container, or assert that the view has the right dimension if A is view
   *
   * @tparam A
   * @param a A container or a view
   */
  template <typename A>
  void resize_or_check_if_view(A &a, std::array<long, A::rank> const &sha) REQUIRES(is_regular_or_view_v<A>) {
    if (a.shape() == sha) return;
    if constexpr (is_regular_v<A>) {
      a.resize(sha);
    } else {
      NDA_RUNTIME_ERROR << "Size mismatch : view class shape = " << a.shape() << " expected " << sha;
    }
  }

#include "./basic_functions.hxx"

} // namespace nda
