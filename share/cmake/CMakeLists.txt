configure_file(nda-config.cmake.in nda-config.cmake @ONLY)
configure_file(nda-config-version.cmake.in nda-config-version.cmake @ONLY)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/nda-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/nda-config-version.cmake
  DESTINATION lib/cmake/nda
)

install(EXPORT nda-targets DESTINATION lib/cmake/nda)
