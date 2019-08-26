#pragma once
namespace nda {

  namespace tag {
    struct regular_or_view {};
    struct regular : regular_or_view {};
    struct view : regular_or_view {};
    struct array_view : view {};
    struct array : regular {};
  
    struct matrix_or_view{};

  } // namespace tag

  template<typename T> inline constexpr bool is_matrix_or_view_v = std::is_base_of_v<tag::matrix_or_view, T>;
  template<typename T> inline constexpr bool is_regular_or_view_v = std::is_base_of_v<tag::regular_or_view, T>;

}
