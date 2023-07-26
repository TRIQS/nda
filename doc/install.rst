.. highlight:: bash

.. _install:

Install nda
***********

Compiling nda from source
=========================

.. note:: To guarantee reproducibility in scientific calculations we strongly recommend the use of a stable `release <https://github.com/TRIQS/triqs/releases>`_ of both TRIQS and its applications.

Installation steps
------------------

#. Download the source code of the latest stable version by cloning the ``TRIQS/nda`` repository from GitHub::

     $ git clone https://github.com/TRIQS/nda nda.src

#. Create and move to a new directory where you will compile the code::

     $ mkdir nda.build && cd nda.build

#. In the build directory call cmake, including any additional custom CMake options, see below::

     $ cmake -DCMAKE_INSTALL_PREFIX=path_to_install_dir ../nda.src

#. Compile the code, run the tests and install the application::

     $ make
     $ make test
     $ make install

Versions
--------

To use a particular version, go into the directory with the sources, and look at all available versions::

     $ cd nda.src && git tag

Checkout the version of the code that you want::

     $ git checkout 1.1.0

and follow steps 2 to 4 above to compile the code.

Linking against nda
-------------------

In order to use nda in other projects you can load it into your current shell environment with::

     $ source path_to_install_dir/share/nda/ndavars.sh

We then recommend to use cmake to link your executable against nda.
This will require the following additional lines in your CMakeLists.txt::

     find_package(nda REQUIRED)
     target_link_libraries(your_executable nda::nda_c)


Custom CMake options
--------------------

The compilation of ``nda`` can be configured using CMake-options::

    cmake ../nda.src -DOPTION1=value1 -DOPTION2=value2 ...

+-------------------------------------------+-----------------------------------------------+
| Options                                   | Syntax                                        |
+===========================================+===============================================+
| Specify an installation path              | -DCMAKE_INSTALL_PREFIX=path_to_nda            |
+-------------------------------------------+-----------------------------------------------+
| Build in Debugging Mode                   | -DCMAKE_BUILD_TYPE=Debug                      |
+-------------------------------------------+-----------------------------------------------+
| Disable Python and Cpp2Py Support         | -DPythonSupport=OFF                           |
+-------------------------------------------+-----------------------------------------------+
| Disable testing (not recommended)         | -DBuild_Tests=OFF                             |
+-------------------------------------------+-----------------------------------------------+
| Build Benchmarks                          | -DBuild_Benchs=ON                             |
+-------------------------------------------+-----------------------------------------------+
| Build the documentation                   | -DBuild_Documentation=ON                      |
+-------------------------------------------+-----------------------------------------------+
