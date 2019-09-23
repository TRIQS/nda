add_library(mpi_c INTERFACE)

target_include_directories(mpi_c
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
)

target_compile_options(mpi_c INTERFACE -std=c++17 $<$<CXX_COMPILER_ID:GNU>:-fconcepts>)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
