add_subdirectory(cmake)

configure_file(nda.modulefile.in nda.modulefile @ONLY)
configure_file(ndavars.sh.in ndavars.sh @ONLY)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/nda.modulefile
    ${CMAKE_CURRENT_BINARY_DIR}/ndavars.sh
  DESTINATION share
)

message(STATUS "***************************************************************")
message(STATUS "* Use:                                                         ")
message(STATUS "*                                                              ")
message(STATUS "*   source ${CMAKE_INSTALL_PREFIX}/share/ndavars.sh      ")
message(STATUS "*                                                              ")
message(STATUS "* to set up the environment variables                          ")
message(STATUS "***************************************************************")
