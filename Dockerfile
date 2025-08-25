# See packaging for various base options
FROM flatironinstitute/triqs:base
ARG APPNAME=app4triqs

# Install here missing dependencies, e.g.
# RUN apt-get install -y python3-skimage

ARG BUILDUID=983
RUN useradd -u $BUILDUID -m build

ENV SRC=/src \
    BUILD=/home/build \
    INSTALL=/usr/local \
    PYTHONPATH=/usr/local/lib/python$PYTHON_VERSION/site-packages \
    CMAKE_PREFIX_PATH=/usr/lib/cmake/$APPNAME \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

COPY --chown=build . $SRC/$APPNAME
RUN mkdir $BUILD/$APPNAME && chown build $BUILD/$APPNAME

ARG BUILD_ID
ARG CMAKE_ARGS
USER build
WORKDIR $BUILD/$APPNAME
ARG NCORES=4
RUN cmake $SRC/$APPNAME -DCMAKE_INSTALL_PREFIX=$INSTALL -DCLANG_OPT="$CXXFLAGS" $CMAKE_ARGS && make -j$NCORES || make -j1 VERBOSE=1
USER root
RUN make install
