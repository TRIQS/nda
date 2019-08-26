#pragma once
namespace nda {

  namespace tag {
    namespace concepts {

      struct ndarray {};                          // General concept for ndarray, but NO algebra defined.
      struct _array : ndarray {};                 // with array algebra
      struct _matrix : ndarray {};                // with matrix algebra
      struct _vector : ndarray {};                // with vector algebra
      struct mutable_ndarray : ndarray {};        // Same with additional condition of being mutable.
      struct mutable_arraty : mutable_ndarray {}; //
      struct mutable_matrix : mutable_ndarray {};
      struct mutable_vector : mutable_ndarray {};

    } // namespace concepts

    namespace containers {

      struct _regular {};
      struct _view {};

      struct _array : _regular {};
      struct _array_view : _view {};

      struct _matrix : _regular {};
      struct _matrix_view : _view {};

      struct _vector : _regular {};
      struct _vector_view : _view {};

    } // namespace containers

  } // namespace tag

  // Recognize the containers

  template <typename T> inline constexpr bool is_regular_v         = std::is_base_of_v<tag::containers::_regular, T>;
  template <typename T> inline constexpr bool is_view_v            = std::is_base_of_v<tag::containers::_view, T>;
  template <typename T> inline constexpr bool is_regular_or_view_v = is_regular_v<T> or is_view_v<T>;

  template <typename T> inline constexpr bool is_array_regular_v = std::is_base_of_v<tag::containers::_array, T>;
  template <typename T> inline constexpr bool is_array_view_v    = std::is_base_of_v<tag::containers::_array_view, T>;

  template <typename T> inline constexpr bool is_matrix_regular_v = std::is_base_of_v<tag::containers::_matrix, T>;
  template <typename T> inline constexpr bool is_matrix_view_v    = std::is_base_of_v<tag::containers::_matrix_view, T>;

  template <typename T> inline constexpr bool is_vector_regular_v = std::is_base_of_v<tag::containers::_vector, T>;
  template <typename T> inline constexpr bool is_vector_view_v    = std::is_base_of_v<tag::containers::_vector_view, T>;

  // Concepts : traits that can be specialized
  template <typename T> struct _is_ndarray : std::is_base_of<tag::concepts::ndarray, T> {};
  template <typename T> struct _is_array : std::is_base_of<tag::concepts::_array, T> {};
  template <typename T> struct _is_matrix : std::is_base_of<tag::concepts::_matrix, T> {};
  template <typename T> struct _is_vector : std::is_base_of<tag::concepts::_vector, T> {};

  //
  template <typename T> inline constexpr bool is_ndarray_v = _is_ndarray<T>::value;
  template <typename T> inline constexpr bool is_array_v   = _is_array<T>::value;
  template <typename T> inline constexpr bool is_matrix_v  = _is_matrix<T>::value;
  template <typename T> inline constexpr bool is_vector_v  = _is_vector<T>::value;

} // namespace nda
