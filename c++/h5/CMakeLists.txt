# ---------------------------------
# HDF5
# ---------------------------------

message(STATUS "-------- HDF5 detection -------------")

# on weiss, it is 2.8.2 and we should not put HL, on 12.04 we need to put it...
if(${CMAKE_VERSION} VERSION_LESS "2.8.6") # CHECK THIS BOUND, where are the cmake changelogs ??
 find_package(HDF5 REQUIRED C)
else(${CMAKE_VERSION} VERSION_LESS "2.8.6")
 find_package(HDF5 REQUIRED C HL)
endif(${CMAKE_VERSION} VERSION_LESS "2.8.6")

if(NOT HDF5_FOUND)
 message(FATAL_ERROR "Require hdf5 1.8.2 or higher. Set HDF5_HOME")
endif()

if(HDF5_IS_PARALLEL)
  message(WARNING "parallel(MPI) hdf5 is detected. The standard version is preferred.")
endif(HDF5_IS_PARALLEL)

if(HDF5_HL_LIBRARIES)    # CMake 3.6.0 and later puts libhdf5_hl into a separate variable
 list(APPEND HDF5_LIBRARIES ${HDF5_HL_LIBRARIES})
 list(REMOVE_DUPLICATES HDF5_LIBRARIES)
endif(HDF5_HL_LIBRARIES)

message(STATUS "HDF5_LIBRARIES = ${HDF5_LIBRARIES}")

# Get hdf5 version and add a definition HDF5_LT_189 if version < 1.8.9
execute_process(COMMAND ${HDF5_DIFF_EXECUTABLE} --version OUTPUT_VARIABLE diff_output)
string(REGEX MATCHALL "[0-9]+.[0-9]+.[0-9]+" HDF5_VERSION "${diff_output}")
message(STATUS "hdf5 version: ${HDF5_VERSION}")

#if(PythonSupport)
 #if(NOT "${PYTHON_H5PY_HDF5VERSION}" STREQUAL "${HDF5_VERSION}")
 #message(FATAL_ERROR "The HDF5 version of the h5py (${PYTHON_H5PY_HDF5VERSION}) is different from the hdf5 library found for C++ (${HDF5_VERSION}). You MUST have the same version.")
 #endif()
#endif()

#add_library(hdf5 INTERFACE)
#target_include_directories(hdf5 SYSTEM INTERFACE ${HDF5_INCLUDE_DIRS})
#target_link_libraries(hdf5 INTERFACE ${HDF5_LIBRARIES})
#target_compile_options(hdf5 INTERFACE ${HDF5_DEFINITIONS})

add_library(__hdf5 INTERFACE)
target_include_directories(__hdf5 SYSTEM INTERFACE ${HDF5_INCLUDE_DIRS})
target_link_libraries(__hdf5 INTERFACE ${HDF5_LIBRARIES})
target_compile_options(__hdf5 INTERFACE ${HDF5_DEFINITIONS})

# -------- Now build our interface -------------------------------


add_library(h5_interface h5object.cpp file.cpp group.cpp
			 array_interface.cpp
                         stl/string.cpp stl/vector.cpp
			 )

target_link_libraries(h5_interface PRIVATE __hdf5 )
target_compile_options(h5_interface PUBLIC -std=c++17)


