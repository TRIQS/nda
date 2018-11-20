file(GLOB_RECURSE sources *.cpp)
add_library(nda_c ${sources})

# Link against triqs and use headers with SYSTEM flag for better warning messages
target_link_libraries(nda_c PUBLIC triqs)
target_include_directories(nda_c SYSTEM PUBLIC ${TRIQS_ROOT}/include)

target_compile_options(nda_c PUBLIC -std=c++17 -fPIC)
target_include_directories(nda_c PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/c++>)
target_compile_definitions(nda_c PRIVATE
				NDA_GIT_HASH=${NDA_GIT_HASH}
			    	TRIQS_GIT_HASH=${TRIQS_GIT_HASH}
				$<$<CONFIG:Debug>:NDA_DEBUG>
			        $<$<CONFIG:Debug>:TRIQS_ARRAYS_ENFORCE_BOUNDCHECK>
			 	)

# Install library and headers
install(TARGETS nda_c EXPORT nda-targets DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include/nda FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")


# ========= Static Analyzer Checks ==========

option(ANALYZE_SOURCES OFF "Run static analyzer checks if found (clang-tidy, cppcheck)")
if(ANALYZE_SOURCES)

  # Locate static analyzer tools
  find_program(CPPCHECK_EXECUTABLE NAMES "cppcheck" PATHS ENV PATH)
  find_program(CLANG_TIDY_EXECUTABLE NAMES "clang-tidy" PATHS ENV PATH)

  # Run clang-tidy if found
  if(CLANG_TIDY_EXECUTABLE)
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXECUTABLE}")
    set_target_properties(nda_c PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
  else()
    message(STATUS "clang-tidy not found in $PATH. Please consider installing clang-tidy for additional checks!")
  endif()

  # Run cppcheck if found
  if(CPPCHECK_EXECUTABLE)
    message(STATUS "cppcheck found: ${CPPCHECK_EXECUTABLE}")
    add_custom_command(
	TARGET nda_c
	COMMAND ${CPPCHECK_EXECUTABLE}
	  --enable=warning,style,performance,portability
	  --std=c++14
	  --template=gcc
	  --verbose
	  --quiet
          ${sources}
    )
  else()
    message(STATUS "cppcheck not found in $PATH. Please consider installing cppcheck for additional checks!")
  endif()

endif()


# ========= Dynamic Analyzer Checks ==========

# Address Sanitizer
option(ASAN OFF "Compile library and executables with LLVM Address Sanitizer")
if(ASAN)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    message(FATAL_ERROR "Address Sanitizer Checks currently not compatible with OSX.")
  endif()
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/share/cmake)
  find_package(libasan_rt REQUIRED)
  target_compile_options(nda_c PUBLIC -fsanitize=address -fno-omit-frame-pointer)
  target_link_libraries(nda_c INTERFACE "-fsanitize=address -fno-omit-frame-pointer -fuse-ld=gold")
endif()

# Undefined Behavior Sanitizer
option(UBSAN OFF "Compile library and executables with LLVM Undefined Behavior Sanitizer")
if(UBSAN)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    message(FATAL_ERROR "Undefined Behavior Sanitizer Checks currently not compatible with OSX.")
  endif()
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/share/cmake)
  find_package(libubsan_rt REQUIRED)
  target_compile_options(nda_c PUBLIC -fsanitize=undefined -fno-omit-frame-pointer -fno-sanitize=vptr)
  target_link_libraries(nda_c INTERFACE "-fsanitize=undefined -fno-omit-frame-pointer -fuse-ld=gold")
endif()
