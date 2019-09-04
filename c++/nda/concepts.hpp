#pragma once
namespace nda {

  namespace tag {

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

BALBLAB A

  template <typename T>
  inline constexpr bool is_regular_v = std::is_base_of_v<tag::containers::_regular, T>;
  template <typename T>
  inline constexpr bool is_view_v = std::is_base_of_v<tag::containers::_view, T>;
  template <typename T>
  inline constexpr bool is_regular_or_view_v = is_regular_v<T> or is_view_v<T>;

  template <typename T>
  inline constexpr bool is_array_regular_v = std::is_base_of_v<tag::containers::_array, T>;
  template <typename T>
  inline constexpr bool is_array_view_v = std::is_base_of_v<tag::containers::_array_view, T>;

  template <typename T>
  inline constexpr bool is_matrix_regular_v = std::is_base_of_v<tag::containers::_matrix, T>;
  template <typename T>
  inline constexpr bool is_matrix_view_v = std::is_base_of_v<tag::containers::_matrix_view, T>;

  template <typename T>
  inline constexpr bool is_vector_regular_v = std::is_base_of_v<tag::containers::_vector, T>;
  template <typename T>
  inline constexpr bool is_vector_view_v = std::is_base_of_v<tag::containers::_vector_view, T>;

} // namespace nda
