add_library(clef_c INTERFACE)

target_include_directories(clef_c
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
)
target_compile_options(clef_c INTERFACE -std=c++17 -fPIC $<$<CXX_COMPILER_ID:GNU>:-fconcepts>)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
