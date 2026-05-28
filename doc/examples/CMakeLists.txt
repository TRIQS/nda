# List of all examples
file(GLOB_RECURSE all_examples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

if(NOT HDF5Support)
  list(FILTER all_examples EXCLUDE REGEX "ex1\\.cpp|ex5\\.cpp|doc_h5_|doc_overview\\.cpp")
endif()

if(NOT MPISupport)
  list(FILTER all_examples EXCLUDE REGEX "ex6\\.cpp|doc_mpi_")
endif()

if(NOT TblisSupport)
  list(FILTER all_examples EXCLUDE REGEX "ex9\\.cpp")
endif()

foreach(ex_src ${all_examples})
  get_filename_component(ex_name ${ex_src} NAME_WE)
  add_executable(${ex_name} ${ex_src})
  target_link_libraries(${ex_name} ${PROJECT_NAME}_c)
endforeach()
