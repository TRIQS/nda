# Fail early if a doc dependency is missing (import names, not distribution names)
find_package(Python COMPONENTS Interpreter REQUIRED)

foreach(_module numpydoc nbsphinx myst_parser linkify_it sphinx_rtd_theme matplotlib)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import ${_module}"
    RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
  if(_rc)
    message(FATAL_ERROR "Building the documentation requires the python module '${_module}' (see requirements.txt).")
  endif()
endforeach()

# Generate the sphinx config file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in ${CMAKE_CURRENT_BINARY_DIR}/conf.py @ONLY)

# --------------------------------------------------------
# Build & Run the C++ doc examples and capture the output
# --------------------------------------------------------
add_custom_target(${PROJECT_NAME}_docs_example_output)
file(GLOB_RECURSE ExampleList RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(example ${ExampleList})
  get_filename_component(f ${example} NAME_WE)
  get_filename_component(d ${example} DIRECTORY)
  add_executable(${PROJECT_NAME}_doc_${f} EXCLUDE_FROM_ALL ${example})
  set_property(TARGET ${PROJECT_NAME}_doc_${f} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${d})
  target_link_libraries(${PROJECT_NAME}_doc_${f} ${PROJECT_NAME}_c)
  add_custom_command(TARGET ${PROJECT_NAME}_doc_${f} POST_BUILD
   COMMAND ${PROJECT_NAME}_doc_${f} > ${CMAKE_CURRENT_SOURCE_DIR}/${d}/${f}.output 2>/dev/null
   WORKING_DIRECTORY ${d}
  )
  add_dependencies(${PROJECT_NAME}_docs_example_output ${PROJECT_NAME}_doc_${f})
endforeach()

# ---------------------------------
# Top Sphinx target
# ---------------------------------
if(NOT DEFINED SPHINXBUILD_EXECUTABLE)
  find_package(Sphinx)
endif()

# Sphinx has internal caching, always run it
add_custom_target(${PROJECT_NAME}_docs_sphinx ALL)
add_dependencies(${PROJECT_NAME}_docs_sphinx ${PROJECT_NAME}_c ${PROJECT_NAME}_python_modules)
add_custom_command(
  TARGET ${PROJECT_NAME}_docs_sphinx POST_BUILD
  COMMAND PYTHONPATH=${PROJECT_BINARY_DIR}/python:$ENV{PYTHONPATH} ${SPHINXBUILD_EXECUTABLE} -j auto -c . -b html ${CMAKE_CURRENT_SOURCE_DIR} html
)

option(Sphinx_Only "When building the documentation, skip the Python Modules and the generation of C++ Api and example outputs" OFF)
if(NOT Sphinx_Only)
  # Generation of C++ Api and Example Outputs
  add_dependencies(${PROJECT_NAME}_docs_sphinx ${PROJECT_NAME}_docs_example_output)
endif()

# ---------------------------------
# Install
# ---------------------------------
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/ COMPONENT documentation DESTINATION share/doc/${PROJECT_NAME}
  FILES_MATCHING
  REGEX "\\.(html|pdf|png|gif|jpg|svg|ico|js|xsl|css|py|txt|inv|bib|ttf|woff2|eot|sh)$"
  PATTERN "_*"
)

# ---------------------------------
# Build doxygen documentation
# ---------------------------------
add_subdirectory(doxygen)
