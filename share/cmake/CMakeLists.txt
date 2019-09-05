configure_file(nda-config.cmake.in nda-config.cmake @ONLY)
configure_file(nda-config-version.cmake.in nda-config-version.cmake @ONLY)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/app4triqs-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/app4triqs-config-version.cmake
  DESTINATION lib/cmake/app4triqs
)

install(EXPORT app4triqs-targets NAMESPACE app4triqs:: DESTINATION lib/cmake/app4triqs)
