# The list of benchss
set(all_benchs view iterator_foreach expr_temp) 

#=== build google benchmark as an external project ===

include(ExternalProject)
set(GBENCH_ROOT ${CMAKE_BINARY_DIR}/gbench)
ExternalProject_Add(gbench_project
  PREFIX gbench
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.4.1
  GIT_CONFIG advice.detachedHead=false
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GBENCH_ROOT}  -DBENCHMARK_ENABLE_GTEST_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
  # Or automatically install gtest and build unittests
  # CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GBENCH_ROOT} -DCMAKE_INSTALL_LIBDIR=lib -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON
)

# Create the imported google benchmark target
add_library(gbench STATIC IMPORTED)
add_dependencies(gbench gbench_project)

set_target_properties(gbench PROPERTIES
  IMPORTED_LOCATION ${GBENCH_ROOT}/lib/libbenchmark.a
  INTERFACE_LINK_LIBRARIES pthread
  ## Populating the INTERFACE_INCLUDE_DIRECTORIES won't work
  ## as the directory does not yet exist
  INTERFACE_COMPILE_OPTIONS "-isystem;${GBENCH_ROOT}/include"
)

#-=======================================


FOREACH( BenchName1  ${all_benchs} )
 STRING(REPLACE ".cpp" "" BenchName ${BenchName1})
 add_executable( ${BenchName}  ${CMAKE_CURRENT_SOURCE_DIR}/${BenchName}.cpp)
 target_link_libraries(${BenchName} nda_c gbench)
ENDFOREACH(BenchName1  ${TestList})


