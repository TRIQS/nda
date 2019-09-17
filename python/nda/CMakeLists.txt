# We MUST make another library and link the module
# since the visibility is different for pybind11 modules (= hidden)
# and it must be the same for the h5_interface and the rest of the h5 files
add_library(h5_interface_python python)
target_link_libraries(h5_interface_python PRIVATE h5_interface __hdf5 python_and_numpy)

# Then the module and link __hdf5_python to it
pybind11_add_module(_h5py _h5py.cpp)
target_link_libraries(_h5py  PRIVATE __hdf5 h5_interface h5_interface_python)  
target_compile_options(_h5py PRIVATE "-Wno-register") # C++17 



## Build the python module
#add_cpp2py_module(toto_module)

#target_link_libraries(toto_module nda_c)

## We need to include the convertes.hxx files
#target_include_directories(toto_module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

## Configure the version
#configure_file(version.py.in version.py)

## All Python files. Copy them in the build dir to have a complete package for the tests.
#set(PYTHON_SOURCES __init__.py)
#foreach(f ${PYTHON_SOURCES})
  #configure_file(${f} ${f} COPYONLY)
#endforeach()

## Install python module to proper location
#set(PYTHON_LIB_DEST ${CPP2PY_PYTHON_LIB_DEST_ROOT}/nda)
#install(TARGETS toto_module DESTINATION ${PYTHON_LIB_DEST})
#install(FILES ${PYTHON_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/version.py DESTINATION ${PYTHON_LIB_DEST})


