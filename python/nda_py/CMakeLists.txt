# Configure the version
configure_file(version.py.in version.py)

find_package(Python3 COMPONENTS Interpreter NumPy)

# ---------------------------------------------------
# nda_py : 
# ---------------------------------------------------
add_library(nda_py numpy_proxy.cpp)
add_library(${PROJECT_NAME}::nda_py ALIAS nda_py)

# Enable warnings
target_link_libraries(nda_py PRIVATE $<BUILD_INTERFACE:${PROJECT_NAME}_warnings>)
target_link_libraries(nda_py PUBLIC nda::nda_c Python3::NumPy)

# Set C++ Standard
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++20" HAS_CPP20)
if(HAS_CPP20)
  target_compile_options(nda_py PUBLIC -std=c++20)
else()
  target_compile_options(nda_py PUBLIC -std=c++17 $<$<CXX_COMPILER_ID:GNU>:-fconcepts>)
endif()

# Configure target and compilation
set_property(TARGET nda_py PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(nda_py PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/python> $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>)

target_include_directories(nda_py SYSTEM INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
target_compile_definitions(nda_py PUBLIC
				NDA_GIT_HASH=${PROJECT_GIT_HASH}
				$<$<CONFIG:Debug>:NDA_DEBUG>
			  )

# Install library and headers
install(TARGETS nda_py EXPORT ${PROJECT_NAME}-targets DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
set(PYTHON_LIB_DEST ${PYTHON_LIB_DEST_ROOT}/nda_py)
install(FILES __init__.py DESTINATION ${PYTHON_LIB_DEST})

