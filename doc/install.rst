.. highlight:: bash

.. _install:

Installation
============


Prerequisite
-------------------

#. The :ref:`TRIQS <triqslibs:welcome>` toolbox (see :ref:`TRIQS installation instruction <triqslibs:installation>`).
   In the following, we will suppose that it is installed in the ``path_to_triqs`` directory.

Installation steps
------------------

#. Download the sources from github::

     $ git clone https://github.com/triqs/nda.git nda.src

#. Create an empty build directory where you will compile the code::

     $ mkdir nda.build && cd nda.build

#. Make sure that you have added the TRIQS and Cpp2Py installation to your environment variables::

     $ source path_to_triqs/share/cpp2pyvarsh.sh
     $ source path_to_triqs/share/triqsvarsh.sh

#. In the build directory call cmake::

     $ cmake ../nda.src

#. Compile the code, run the tests and install the application::

     $ make
     $ make test
     $ make install

Version compatibility
---------------------

Be careful that the version of the TRIQS library and of the solver must be
compatible (more information on the :ref:`TRIQS website <triqslibs:versions>`).
As nda is still in alpha phase (unstable), it can only be compiled against the
unstable branch of triqs.

Custom CMake options
--------------------

Functionality of ``nda`` can be tweaked using extra compile-time options passed to CMake::

    cmake -DOPTION1=value1 -DOPTION2=value2 ... ../nda.src

+-----------------------------------------------------------------------+-----------------------------------------------+
| Options                                                               | Syntax                                        |
+=======================================================================+===============================================+
| Specify an installation path other than path_to_triqs                 | -DCMAKE_INSTALL_PREFIX=path_to_nda      |
+-----------------------------------------------------------------------+-----------------------------------------------+
| Build the documentation locally                                       | -DBuild_Documentation=ON                      |
+-----------------------------------------------------------------------+-----------------------------------------------+
