add_subdirectory(cmake)

if(NOT IS_SUBPROJECT)

  if(PythonSupport)
    set(EXPORT_PYTHON_PATH "export PYTHONPATH=${CMAKE_INSTALL_PREFIX}/${CPP2PY_PYTHON_LIB_DEST_ROOT}:$PYTHONPATH")
    set(MODFILE_PYTHON_PATH "prepend-path    PYTHONPATH          $root/${CPP2PY_PYTHON_LIB_DEST_ROOT}")
  endif()

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
  message(STATUS "*   source ${CMAKE_INSTALL_PREFIX}/share/ndavars.sh            ")
  message(STATUS "*                                                              ")
  message(STATUS "* to set up the environment variables                          ")
  message(STATUS "***************************************************************")

endif()
