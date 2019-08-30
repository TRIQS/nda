 // UNUSED
  //------------------------------------------------------------
  // get_call_const_return_t : alias that gives the type of a(0,0,0...) with A.rank zeros.
  template <size_t... Is, typename A> void __call_with_0(std::index_sequence<Is...>, A &&a) {
    return a((0, std::get<Is>)...); // repeat 0 sizeof...(Is) times
  }

  template <typename A> struct __get_call_const_return_type {
    using type = decltype(__call_with_0(std::make_index_sequence<A::rank()>, std::declval<A const>));
  };
  template <typename A> using get_call_const_return_t = __get_call_const_return_type<A>::type;

 
