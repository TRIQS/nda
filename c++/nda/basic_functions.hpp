#pragma once

namespace nda {

  // --------------------------- make_regular ------------------------
  // general make_regular
  // FIXME : auto return ?  regular_t<A> ?
  template <typename A>
  basic_array<get_value_t<std::decay_t<A>>, get_rank<A>, C_layout, get_algebra<std::decay_t<A>>, heap> //
  make_regular(A &&x) REQUIRES(is_ndarray_v<std::decay_t<A>>) {
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
