#pragma once

namespace nda {

  template <typename V>
  matrix<V> eye(long dim) {
    matrix<V> r(dim, dim);
    r = 1;
    return r;
  }

  /* template <typename ArrayType>
  matrix_view<typename ArrayType::value_type> make_matrix_view(ArrayType const &a) {
    static_assert(ArrayType::rank == 2, "make_matrix_view only works for array of rank 2");
    return a;
  }

  template <typename ArrayType>
  matrix<typename ArrayType::value_type> make_matrix(ArrayType const &a) {
    static_assert(ArrayType::domain_type::rank == 2, "make_matrix only works for array of rank 2");
    return a;
  }
*/

#if __cplusplus > 201703L
  template <Array<2> M>
#else
  template <typename M>
#endif
  typename M::value_type trace(M const &m) NDA_REQUIRES17(is_ndarray_v<M> and (get_rank<M> == 2))
  {
    EXPECTS(m.extent(0) == m.extent(1));
    auto r = typename M::value_type{};
    auto d = m.extent(0);
    for (int i = 0; i < d; ++i) r += m(i, i);
    return r;
  }

  ///
  template <typename M>
  auto dagger(M const &m) NDA_REQUIRES(is_ndarray_v<M> and (get_rank<M> == 2)) {
    return conj(transpose(m));
  }

  ///
  template <typename M1, typename M2>
  matrix<typename M1::value_type> vstack(M1 const &a, M2 const &b) {
    static_assert(std::is_same_v<typename M1::value_type, typename M2::value_type>, "ERROR in vstack(A,B): Matrices have incompatible value_type!");
    EXPECTS_WITH_MESSAGE(a.extent(1) == b.extent(1), "ERROR in vstack(A,B): Matrices have incompatible shape!");

    matrix<typename M1::value_type> res(a.extent(0) + b.extent(0), a.extent(1));
    res(range(a.extent(0)), range_all{})                            = a;
    res(range(a.extent(0), a.extent(0) + b.extent(0)), range_all{}) = b;
    return res;
  }

} // namespace nda
