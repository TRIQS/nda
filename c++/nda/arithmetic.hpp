#pragma once
#include "linalg/matmul.hpp"

namespace nda {

  // Scalar matrix and array
  template <typename S, int Rank>
  struct scalar_array;

  template <typename S>
  struct scalar_matrix;

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

  template <typename S, int Rank>
  inline constexpr bool is_ndarray_v<scalar_array<S, Rank>> = true;

  template <typename S>
  inline constexpr bool is_ndarray_v<scalar_matrix<S>> = true;

  // Get the layout info recursively
  template <char OP, typename L, typename R>
  inline constexpr layout_info_t get_layout_info<expr<OP, L, R>> = expr<OP, L, R>::layout_info;

  template <char OP, typename L>
  inline constexpr layout_info_t get_layout_info<expr_unary<OP, L>> = get_layout_info<std::decay_t<L>>;

  template <typename S, int Rank>
  inline constexpr layout_info_t get_layout_info<scalar_array<S, Rank>> = layout_info_t{0, layout_prop_e::none};

  template <typename S>
  inline constexpr layout_info_t get_layout_info<scalar_matrix<S>> = layout_info_t{}; // NOT contiguous, we disable the linear optimisation of assign

  // -------------------------------------------------------------------------------------------
  //                         A simple type for scalar_matrix, scalar_array, scalar_vector
  // -------------------------------------------------------------------------------------------

  template <typename S, int Rank>
  struct scalar_array {
    S const s;
    shape_t<Rank> _shape;

    [[nodiscard]] shape_t<Rank> const &shape() const { return _shape; }

    template <typename T>
    S operator[](T &&) const {
      return s;
    }

    template <typename... T>
    S operator()(T &&...) const {
      return s;
    }
  };

  //----------------------

  template <typename S>
  struct scalar_matrix {
    S const s;

    shape_t<2> _shape;

    [[nodiscard]] shape_t<2> const &shape() const { return _shape; }

    template <typename T>
    S operator[](T &&) const {
      return s;
    }
    template <typename A1, typename A2>
    S operator()(A1 const &a1, A2 const &a2) const {
      return (a1 == a2 ? s : S{});
    }
  };

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

    static constexpr layout_info_t layout_info =
       (l_is_scalar ? get_layout_info<R_t> : (r_is_scalar ? get_layout_info<L_t> : get_layout_info<R_t> | get_layout_info<L_t>));

    [[nodiscard]] constexpr auto shape() const {
      if constexpr (l_is_scalar) {
        return r.shape();
      } else if constexpr (r_is_scalar) {
        return l.shape();
      } else {
        EXPECTS(l.shape() == r.shape());
        return l.shape();
      }
    }

    private:
    FORCEINLINE static bool kronecker(long i, long j) { return (i == j); }

    public:
    // FIXME Clef
    template <typename... Args>
    auto operator()(Args const &... args) const { //  REQUIRES(not(clef::is_lazy<A> and ...)) {

      // We simply implement all cases
      if constexpr (OP == '+') { // we KNOW that l and r are NOT scalar (cf operator +,- impl).
        return l(args...) + r(args...);
      }

      if constexpr (OP == '-') { return l(args...) - r(args...); }

      if constexpr (OP == '*') {
        if constexpr (l_is_scalar)
          return l * r(args...);
        else if constexpr (r_is_scalar)
          return l(args...) * r;
        else {
          static_assert(algebra != 'M', "Should not occur");
          return l(args...) * r(args...);
        }
      }

      if constexpr (OP == '/') {
        //static_assert(algebra != 'M', "Should not occur");
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

    private: // detail for operator [] below
    static constexpr bool rank_is_one() {
      if constexpr (nda::is_scalar_v<L_t> || nda::is_scalar_v<R_t>)
        return true;
      else
        return ((get_rank<L_t> == 1) and (get_rank<R_t> == 1));
    }

    public:
    // FIXME
    // [long] ? 1d only ? strided only ?
    // Overload with _long ? long ? lazy ?
    /// [ ] is the same as (). Enable for Vectors only
    template <typename Arg>
    auto operator[](Arg const &arg) const {
      static_assert(rank_is_one(), "operator[] only available for array of rank 1");
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
    auto operator()(Args &&... args) const {                           // REQUIRES(not(clef::is_lazy<L>))
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

  // checks that L, R model NdArray, with possible one being a scalar
  template <typename L, typename R>
  constexpr bool
     model_ndarray_with_possibly_one_scalar = (is_ndarray_v<std::decay_t<L>> and (is_ndarray_v<std::decay_t<R>> or is_scalar_v<std::decay_t<R>>))
     or (is_scalar_v<std::decay_t<L>> and is_ndarray_v<std::decay_t<R>>);

  // checks that ranks are the same
  template <typename L, typename R>
  constexpr bool rank_are_compatible() {
    using L_t = std::decay_t<L>; // L, R can be lvalue references
    using R_t = std::decay_t<R>;
    if constexpr (nda::is_scalar_v<L_t> || nda::is_scalar_v<R_t>)
      return true;
    else
      return (get_rank<L_t> == get_rank<R_t>);
  }

  // return 'A', 'M', 'N' (for None), the common algebra of L, R
  // ignoring the scalar type.
  template <typename L, typename R>
  constexpr char common_algebra() {
    using L_t = std::decay_t<L>; // L, R can be lvalue references
    using R_t = std::decay_t<R>;
    if constexpr (nda::is_scalar_v<L_t>) return get_algebra<R_t>;
    if constexpr (nda::is_scalar_v<R_t>) return get_algebra<L_t>;
    if constexpr (get_algebra<L_t> == get_algebra<R_t>)
      return get_algebra<R_t>;
    else
      return 'N';
  }

  /**
   * @tparam L
   * @tparam R
   * @param l : lhs
   * @param r : rhs
   * @requires L, R model NdArray. One can be a scalar. They must be in the same algebra. 
   * @return a lazy expression for elementwise addition
   */
  template <typename L, typename R>
  auto operator+(L &&l, R &&r) REQUIRES(model_ndarray_with_possibly_one_scalar<L, R>) {
    static_assert(rank_are_compatible<L, R>(), "rank mismatch in array addition");
    static_assert(common_algebra<L, R>() != 'N', "Can not add two objects belonging to different algebras");
    using L_t = std::decay_t<L>; // L, R can be lvalue references
    using R_t = std::decay_t<R>;

    if constexpr (is_scalar_v<L_t>) {
      if constexpr (get_algebra<R_t> == 'M')
        return expr<'+', scalar_matrix<L_t>, R>{scalar_matrix<L_t>{l, r.shape()}, std::forward<R>(r)};
      else
        return expr<'+', scalar_array<L_t, R_t::rank>, R>{{l, r.shape()}, std::forward<R>(r)};
    } //
    else if constexpr (is_scalar_v<R_t>) {
      if constexpr (get_algebra<L_t> == 'M')
        return expr<'+', L, scalar_matrix<R_t>>{std::forward<L>(l), scalar_matrix<R_t>{r, l.shape()}};
      else
        return expr<'+', L, scalar_array<R_t, L_t::rank>>{std::forward<L>(l), {r, l.shape()}};
    } //
    else
      return expr<'+', L, R>{std::forward<L>(l), std::forward<R>(r)};
  }

  /**
   * @tparam L
   * @tparam R
   * @param l : lhs
   * @param r : rhs
   * @requires L, R model NdArray. One can be a scalar. They must be in the same algebra. 
   * @return a lazy expression for elementwise substraction
   */
  template <typename L, typename R>
  auto operator-(L &&l, R &&r) REQUIRES(model_ndarray_with_possibly_one_scalar<L, R>) {
    static_assert(rank_are_compatible<L, R>(), "rank mismatch in array addition");
    static_assert(common_algebra<L, R>() != 'N', "Can not substract two objects belonging to different algebras");
    using L_t = std::decay_t<L>; // L, R can be lvalue references
    using R_t = std::decay_t<R>;

    if constexpr (is_scalar_v<L_t>) {
      if constexpr (get_algebra<R_t> == 'M')
        return expr<'-', scalar_matrix<L_t>, R>{scalar_matrix<L_t>{l, r.shape()}, std::forward<R>(r)};
      else
        return expr<'-', scalar_array<L_t, R_t::rank>, R>{{l, r.shape()}, std::forward<R>(r)};
    } //
    else if constexpr (is_scalar_v<R_t>) {
      if constexpr (get_algebra<L_t> == 'M')
        return expr<'-', L, scalar_matrix<R_t>>{std::forward<L>(l), scalar_matrix<R_t>{r, l.shape()}};
      else
        return expr<'-', L, scalar_array<R_t, L_t::rank>>{std::forward<L>(l), {r, l.shape()}};
    } //
    else
      return expr<'-', L, R>{std::forward<L>(l), std::forward<R>(r)};
  }

  /**
   * @tparam L
   * @tparam R
   * @param l : lhs
   * @param r : rhs
   * @requires L, R model NdArray. One can be a scalar. They must be in the same algebra. 
   *    * if the algebra is 'A' : lazy expression for element-wise multiplication
   *    * if the algebra is 'M' : compute the matrix product (with blas gemm), in a new matrix. 
   */
  template <typename L, typename R>
  auto operator*(L &&l, R &&r)REQUIRES(model_ndarray_with_possibly_one_scalar<L, R>) {
    static_assert(rank_are_compatible<L, R>(), "rank mismatch in multiplication");
    static_assert(common_algebra<L, R>() != 'N', "Can not multiply two objects belonging to different algebras");
    using L_t = std::decay_t<L>; // L, R can be lvalue references
    using R_t = std::decay_t<R>;

    // first case : Array algebra or matrix * scalar : expr template
    if constexpr ((common_algebra<L, R>() == 'A') or is_scalar_v<L_t> or is_scalar_v<R_t>) { // array
#ifdef NDA_DEBUG
      if constexpr (!nda::is_scalar_v<L_t> && !nda::is_scalar_v<R_t>)
        if (l.shape() != r.shape()) NDA_RUNTIME_ERROR << "Matrix product : dimension mismatch in matrix product " << l.shape() << " " << r.shape();
#endif
      return expr<'*', L, R>{std::forward<L>(l), std::forward<R>(r)};
    } else {
      return matmul(std::forward<L>(l), std::forward<R>(r));
    }
  }

  /**
   * @tparam L
   * @tparam R
   * @param l : lhs
   * @param r : rhs
   * @requires L, R model NdArray. One can be a scalar. They must be in the same algebra. 
   *    * if the algebra is 'M' for L, then R must be a scalar. matrix/matrix is disabled. 
   *      NB : we could rewrite it as matrix * inverse(matrix) as in triqs arrays, but this looks ambigous.
   * @return lazy expression for element-wise division
   */
  template <typename L, typename R>
  expr<'/', L, R> operator/(L &&l, R &&r) REQUIRES(model_ndarray_with_possibly_one_scalar<L, R> and (common_algebra<L, R>() != 'N')) {
    static_assert(rank_are_compatible<L, R>(), "rank mismatch in array addition");
    static_assert(common_algebra<L, R>() != 'N', "Can not divide two objects belonging to different algebras");
    //using L_t = std::decay_t<L>; // L, R can be lvalue references
    using R_t = std::decay_t<R>;
    static_assert((not(get_algebra<R_t> == 'M') or is_scalar_v<R_t>), " matrix/matrix is not permitted. Use matrix * inverse( ...)");
    return {std::forward<L>(l), std::forward<R>(r)};
  }

  template <typename L>
  expr_unary<'-', L> operator-(L &&l) REQUIRES(is_ndarray_v<std::decay_t<L>>) {
    return {std::forward<L>(l)};
  }

  //------------  lazy inverse

  template <class A>
  expr<'/', A, int> inverse(A &&a) REQUIRES(is_ndarray_v<std::decay_t<A>> and (get_algebra<std::decay_t<A>> != 'M')) {
    return {1, std::forward<A>(a)};
  }

  //------------  Inverse of Matrix

  //// anything / matrix ---> anything * inverse(matrix)
  //template <typename A, typename M>
  //auto operator/(A &&a, M &&m)    //
  //REQUIRES(ImmutableMatrix<M>) //
  //{
  //return std::forward<A>(a) * inverse(std::forward<M>(m);
  //}

} // namespace nda
