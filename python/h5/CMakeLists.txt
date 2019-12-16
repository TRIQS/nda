# Configure the version
configure_file(version.py.in version.py)

# All Python files. Copy them in the build dir to have a complete package for the tests.
file(GLOB_RECURSE python_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py)
foreach(file ${python_sources})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# Build the pybind11 module
pybind11_add_module(_h5py _h5py.cpp h5py_io.cpp)
target_link_libraries(_h5py PRIVATE pybind11 h5_interface hdf5 python_and_numpy)

## Install python modules to proper location
#set(PYTHON_LIB_DEST ${TRIQS_PYTHON_LIB_DEST_ROOT}/h5)
#install(TARGETS _h5py DESTINATION ${PYTHON_LIB_DEST})
#install(FILES ${python_sources} ${CMAKE_CURRENT_BINARY_DIR}/version.py DESTINATION ${PYTHON_LIB_DEST})
