// ADD vector functions

template <typename V>
matrix<V> make_unit_matrix(int dim) {
  matrix<V> r(dim, dim);
  r() = 1;
  return r;
}

template <typename ArrayType>
matrix_view<typename ArrayType::value_type> make_matrix_view(ArrayType const &a) {
  static_assert(ArrayType::rank == 2, "make_matrix_view only works for array of rank 2");
  return a;
}

template <typename ArrayType>
matrix<typename ArrayType::value_type> make_matrix(ArrayType const &a) {
  static_assert(ArrayType::domain_type::rank == 2, "make_matrix only works for array of rank 2");
  return a;
}

template <typename M>
TYPE_ENABLE_IF(typename M::value_type, ImmutableMatrix<M>)
trace(M const &m) {
  auto r = typename M::value_type{};
  if (first_dim(m) != second_dim(m)) TRIQS_RUNTIME_ERROR << " Trace of a non square matrix";
  auto d = first_dim(m);
  for (int i = 0; i < d; ++i) r += m(i, i);
  return r;
}

template <typename M>
std::enable_if_t<ImmutableMatrix<M>::value and triqs::is_complex<typename M::value_type>::value, matrix<typename M::value_type>> dagger(M const &m) {
  return conj(m.transpose());
}

template <typename M>
std::enable_if_t<ImmutableMatrix<M>::value and !triqs::is_complex<typename M::value_type>::value, matrix<typename M::value_type>> dagger(M const &m) {
  return m.transpose();
}

template <typename M1, typename M2>
std::enable_if_t<ImmutableMatrix<M1>::value && ImmutableMatrix<M2>::value, matrix<typename M1::value_type>> vstack(M1 const &A, M2 const &B) {
  TRIQS_ASSERT2(second_dim(A) == second_dim(B), "ERROR in vstack(A,B): Matrices have incompatible shape!");
  static_assert(std::is_same_v<typename M1::value_type, typename M2::value_type>, "ERROR in vstack(A,B): Matrices have incompatible value_type!");

  matrix<typename M1::value_type> res(first_dim(A) + first_dim(B), second_dim(A));
  res(range(first_dim(A)), range())                              = A;
  res(range(first_dim(A), first_dim(A) + first_dim(B)), range()) = B;
  return std::move(res);
}

/
