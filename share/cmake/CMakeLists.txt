# Was the NVHPC SDK used for the CUDA libraries at build time?
if(NVHPC_DIR)
  set(NVHPC_FOUND TRUE)
else()
  set(NVHPC_FOUND FALSE)
endif()

configure_file(${PROJECT_NAME}-config.cmake.in ${PROJECT_NAME}-config.cmake @ONLY)
configure_file(${PROJECT_NAME}-config-version.cmake.in ${PROJECT_NAME}-config-version.cmake @ONLY)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(EXPORT ${PROJECT_NAME}-targets NAMESPACE ${PROJECT_NAME}:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
