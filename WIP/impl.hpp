namespace nda::details {
  // NO EXCEPT ? if stogage is [],

  template <typename... T> constexpr bool ellipsis_is_present = ((std::is_same_v<T, ellipsis> ? 1 : 0) + ...);

  template <typename Self, typename... Args> static FORCEINLINE decltype(auto) _call_(Self &&self, Args const &... args) noexcept {

    using self_t = std::decay_t<Self>;
    if constexpr (sizeof...(Args) == 0) return self_t::const_view_t{*this};

    //if constexpr (clef::is_any_lazy_v<Args...>) { // Is it a lazy call ?
    //if constexpr (R >= 0) static_assert(Number_of_Arguments == R, "Incorrect number of parameters in call");
    //return make_expr_call(std::forward<Self>(self), std::forward<Args>(args)...);
    //} else {                  // not lazy
    // FIXME : Clean this else, return before ?
    if constexpr (self_t::rank >= 0) {
      static_assert((Number_of_Arguments == R) or (ellipsis_is_present<Args...> and (Number_of_Arguments <= R)), "Incorrect number of parameters in call");
    }
    auto idx_sliced_or_position = _idx_m.slice(args...);      // we call the index map
    if constexpr (std::is_same_v<decltype(idx_sliced), long>) // Case 1: we got a long, hence access a element
      return _storage[idx_sliced];
    else // Case 2: we got a slice
      return self_t::view_t{std::move(idx_sliced), _storage};
    //}
  }

} // namespace nda::details

