# Configure the version
configure_file(version.py.in version.py)

# All Python files. Copy them in the build dir to have a complete package for the tests.
file(GLOB_RECURSE python_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py)
foreach(file ${python_sources})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# Install python files to proper location
set(PYTHON_LIB_DEST ${PYTHON_LIB_DEST_ROOT}/${PROJECT_NAME})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.py DESTINATION ${PYTHON_LIB_DEST})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${PYTHON_LIB_DEST_ROOT} FILES_MATCHING PATTERN "*.py")

# Add an interface target for all Python modules
add_library(${PROJECT_NAME}_python_modules INTERFACE)

# Build and install any python modules
file(GLOB_RECURSE module_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(module_src ${module_sources})
  get_filename_component(module_name ${module_src} NAME_WE)
  get_filename_component(module_dir ${module_src} DIRECTORY)

  c2py_add_module(${module_name}
    DIR ${module_dir}
    LINK_LIBRARIES ${PROJECT_NAME}_c ${PROJECT_NAME}_warnings)
  add_library(${PROJECT_NAME}::${module_name} ALIAS ${module_name})
  add_dependencies(${PROJECT_NAME}_python_modules ${module_name})
  install(TARGETS ${module_name} DESTINATION ${PYTHON_LIB_DEST}/${module_dir})
endforeach()
