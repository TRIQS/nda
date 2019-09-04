#pragma once
namespace nda {

  // binary expression
  template <char OP, typename L, typename R>
  struct expr;
  
  // unary expression
  template <char OP, typename L>
  struct expr_unary;

  // algebra 
  template <char OP, typename L, typename R>
  constexpr char get_algebra<expr<OP, L, R>> = expr<OP, L, R>::algebra;

  template <char OP, typename L>
  inline constexpr bool get_algebra<expr_unary<OP, L>> = expr_unary<OP, L>::algebra;

  // Both model NdArray concept
  template <char OP, typename L, typename R>
  inline constexpr bool is_ndarray_v<expr<OP, L, R>> = true;

  template <char OP, typename L>
  inline constexpr bool is_ndarray_v<expr_unary<OP, L>> = true;

  // Both propagate the guarantees
  template <char OP, typename L, typename R>
  inline constexpr uint64_t get_guarantee<expr<OP, L, R>> = get_guarantee<L> &get_guarantee<R>;

  template <char OP, typename L>
  inline constexpr uint64_t get_guarantee<expr_unary<OP, L>> = get_guarantee<L>;

  // true iif rank or L and R is one (or they are scalar)
  template <typename L, typename R>
  constexpr bool rank_is_one() {
    using L_t = std::decay_t<L>; // L, R can be lvalue references
    using R_t = std::decay_t<R>;
    if constexpr (nda::is_scalar_v<L_t> || nda::is_scalar_v<R_t>)
      return true;
    else
      return ((get_rank<L_t> == 1) and (get_rank<R_t> == 1));
  }

  // -------------------------------------------------------------------------------------------
  //                             binary expressions
  // -------------------------------------------------------------------------------------------
  // OP : '+', '-', ...
  template <char OP, typename L, typename R>
  struct expr {

    L l;
    R r;

    using L_t = std::decay_t<L>; // L, R can be lvalue references
    using R_t = std::decay_t<R>;

    static constexpr bool l_is_scalar = nda::is_scalar_v<L_t>;
    static constexpr bool r_is_scalar = nda::is_scalar_v<R_t>;

    static constexpr char algebra = (l_is_scalar ? get_algebra<R_t> : get_algebra<L_t>);

    constexpr auto shape() const {
      if constexpr (l_is_scalar) {
        return r.shape();
      } else if constexpr (r_is_scalar) {
        return l.shape();
      } else {
        EXPECTS(l.shape() == r.shape());
        return l.shape();
      }
    }

    //// REALLY NEEDED ?
    //template <typename LL, typename RR>
    //expr(LL &&l_, RR &&r_) : l(std::forward<LL>(l_)), r(std::forward<RR>(r_)) {}

    // FIXME Clef
    template <typename... Args>
    auto operator()(Args const &... args) const { //  REQUIRES(not(clef::is_lazy<A> and ...)) {

      // We simply implement all cases
      if constexpr (OP == '+') {
        if constexpr (l_is_scalar) {                               // scalar + nda
          if constexpr (algebra != 'M')                            // matrix is a special case
            return l + r(args...);                                 // simply add the scalar
          else                                                     // matrix case: 1 is Identity according to matrix algebra and args is of size 2
            return (operator==(args...) ? l : L_t{}) + r(args...); // L_t{} is the zero of the type
        } else if constexpr (r_is_scalar) {                        // same thing with R
          if constexpr (algebra != 'M')
            return l(args...) + r;
          else
            return l(args...) + (operator==(args...) ? r : R_t{});
        } else
          return l(args...) + r(args...); // generic case, simply add
      }

      if constexpr (OP == '-') { // same as for + with obvious change
        if constexpr (l_is_scalar) {
          if constexpr (algebra != 'M')
            return l - r(args...);
          else // matrix case: 1 is Identity according to matrix algebra and args is of size 2
            return (operator==(args...) ? l : L_t{}) - r(args...);
        } else if constexpr (r_is_scalar) {
          if constexpr (algebra != 'M')
            return l(args...) - r;
          else
            return l(args...) - (operator==(args...) ? r : R_t{});
        } else
          return l(args...) - r(args...);
      }

      if constexpr (OP == '*') {
        static_assert(algebra != 'M', "Should not occur");
        if constexpr (l_is_scalar)
          return l * r(args...);
        else if constexpr (r_is_scalar)
          return l(args...) * r;
        else
          return l(args...) * r(args...);
      }

      if constexpr (OP == '/') {
        static_assert(algebra != 'M', "Should not occur");
        if constexpr (l_is_scalar)
          return l / r(args...);
        else if constexpr (r_is_scalar)
          return l(args...) / r;
        else
          return l(args...) / r(args...);
      }
    }

    // FIXME clef
    //TRIQS_CLEF_IMPLEMENT_LAZY_CALL(); // can not simply capture in () and dispatch becuase of && case. Cf macro def.

    // FIXME 
    // [long] ? 1d only ? strided only ?
    // Overload with _long ? long ? lazy ?
    /// [ ] is the same as (). Enable for Vectors only
    template <typename Arg>
    auto operator[](Arg const &arg) const REQUIRES(rank_is_one<L, R>()) {
      return operator()(std::forward<Arg>(arg));
    }

    // just for better error messages
    template <typename T>
    void operator=(T &&x) = delete; // expressions are immutable
    template <typename T>
    void operator+=(T &&x) = delete; // expressions are immutable
    template <typename T>
    void operator-=(T &&x) = delete; // expressions are immutable
  };

  // -------------------------------------------------------------------------------------------
  //                             unary expressions
  // -------------------------------------------------------------------------------------------

  template <char OP, typename L>
  struct expr_unary {
    using L_t = std::decay_t<L>;
    L l;
    static constexpr char algebra = L_t::algebra;

    template <typename LL>
    expr_unary(LL &&l_) : l(std::forward<LL>(l_)) {}

    // FIXME clef
    template <typename... Args>
    auto operator()(Args &&... args) const {                           // REQUIRES(not(clef::is_lazy<L>)) {
      if constexpr (OP == '-') return -l(std::forward<Args>(args)...); // other cases not implemented
    }

    //    TRIQS_CLEF_IMPLEMENT_LAZY_CALL();

    constexpr auto shape() const { return l.shape(); }

    // just for better error messages
    template <typename T>
    void operator=(T &&x) = delete; // expressions are immutable
    template <typename T>
    void operator+=(T &&x) = delete; // expressions are immutable
    template <typename T>
    void operator-=(T &&x) = delete; // expressions are immutable
  };

  // -------------------------------------------------------------------------------------------
  //                                 Operator overload
  // -------------------------------------------------------------------------------------------

  template <typename L, typename R>
  constexpr bool _ok_for_op = (is_ndarray_v<std::decay_t<L>> and (is_ndarray_v<std::decay_t<R>> or is_scalar_v<std::decay_t<R>>))
     or (is_scalar_v<std::decay_t<L>> and is_ndarray_v<std::decay_t<R>>);

  template <typename L, typename R>
  constexpr bool rank_are_compatible() {
    using L_t = std::decay_t<L>; // L, R can be lvalue references
    using R_t = std::decay_t<R>;
    if constexpr (nda::is_scalar_v<L_t> || nda::is_scalar_v<R_t>)
      return true;
    else
      return (get_rank<L_t> == get_rank<R_t>);
  }

  template <typename L, typename R>
  constexpr bool algebra_are_compatible() {
    using L_t = std::decay_t<L>; // L, R can be lvalue references
    using R_t = std::decay_t<R>;
    if constexpr (nda::is_scalar_v<L_t> || nda::is_scalar_v<R_t>)
      return true;
    else
      return (get_algebra<L_t> == get_algebra<R_t>);
  }

  // FIXME : Fixm error message with tests are ok
  /**
   * Add two array expressions
   * @requires : L or R is a ndaarray
   */
  template <typename L, typename R>
  expr<'+', L, R> operator+(L &&l, R &&r) REQUIRES(_ok_for_op<L, R>) {
    static_assert(rank_are_compatible<L, R>(), "rank mismatch in array addition");
    static_assert(algebra_are_compatible<L, R>(), "Can not add two objects belonging to different algebras");
    return {std::forward<L>(l), std::forward<R>(r)};
  }

  template <typename L, typename R>
  expr<'-', L, R> operator-(L &&l, R &&r) REQUIRES(_ok_for_op<L, R>) {
    static_assert(rank_are_compatible<L, R>(), "rank mismatch in array addition");
    static_assert(algebra_are_compatible<L, R>(), "Can not add two objects belonging to different algebras");
    return {std::forward<L>(l), std::forward<R>(r)};
  }

  template <typename L, typename R>
  auto operator*(L &&l, R &&r)REQUIRES(_ok_for_op<L, R>) {
    static_assert(rank_are_compatible<L, R>(), "rank mismatch in array addition");
    static_assert(algebra_are_compatible<L, R>(), "Can not add two objects belonging to different algebras");
    return expr<'*', L, R>{std::forward<L>(l), std::forward<R>(r)};
  }

  template <typename L, typename R>
  expr<'/', L, R> operator/(L &&l, R &&r) REQUIRES(_ok_for_op<L, R> and (get_algebra<L> != 'M')) {
    static_assert(rank_are_compatible<L, R>(), "rank mismatch in array addition");
    static_assert(algebra_are_compatible<L, R>(), "Can not add two objects belonging to different algebras");
    return {std::forward<L>(l), std::forward<R>(r)};
  }

  template <typename L>
  expr_unary<'-', L> operator-(L &&l) REQUIRES(is_ndarray_v<std::decay_t<L>>) {
    return {std::forward<L>(l)};
  }

  //------------  lazy inverse

  template <class A>
  expr<'/', A, int> inverse(A &&a) REQUIRES(is_ndarray_v<A> and (get_algebra<A> != 'M')) {
    return {1, std::forward<A>(a)};
  }

  //------------  make_array

  ///// Makes a new regular object (array, matrix, vector) from the expression
  //template <char OP, typename L, typename R> array<value_type, expr<OP, L, R>::rank()> make_array(expr<OP, L, R> const &e) { return e; }

  ///// Makes a new regular object (array, matrix, vector) from the expression
  //template <typename L> array<value_type, expr_unary<L>::rank()> make_array(expr_unary<L> const &e) { return e; }

  ///// Makes a new regular object (array, matrix, vector) from the expression
  //template <char OP, typename L, typename R> auto make_regular(expr<OP, L, R> const &x) {
  //return make_array(x);
  //}

  //------------  Inverse of Matrix

  //// anything / matrix ---> anything * inverse(matrix)
  //template <typename A, typename M>
  //auto operator/(A &&a, M &&m)    //
  //REQUIRES(ImmutableMatrix<M>) //
  //{
  //return std::forward<A>(a) * inverse(std::forward<M>(m);
  //}

} // namespace nda
