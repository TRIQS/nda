add_subdirectory(cmake)

if(NOT CMAKE_INSTALL_PREFIX STREQUAL TRIQS_ROOT)

  configure_file(nda.modulefile.in nda.modulefile @ONLY)
  configure_file(ndavars.sh.in ndavars.sh @ONLY)

  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/nda.modulefile
      ${CMAKE_CURRENT_BINARY_DIR}/ndavars.sh
    DESTINATION share
  )
 
  message(STATUS "***************************************************************")
  message(STATUS "* Custom install Location. Use: 			         ")
  message(STATUS "*                                                              ")
  message(STATUS "*   source ${CMAKE_INSTALL_PREFIX}/share/ndavars.sh      ")
  message(STATUS "*                                                              ") 
  message(STATUS "* to set up the environment variables                          ") 
  message(STATUS "***************************************************************")

endif()
