FROM dunecommunity/testing-base_debian:latest

# COMPILER Selection (ARGS cannot be uppercase)
ARG cc
ARG cxx
ARG commit
ENV CC=$cc \
    CXX=$cxx \
    MY_MODULE=dune-gdt

USER dune-ci
ENV HOME=/home/dune-ci
RUN git clone https://github.com/dune-community/dune-gdt-super.git ${SUPERDIR} && \
    cd ${SUPERDIR} && \
    git checkout ${commit} && \
    git submodule foreach git reset --hard && \
    git submodule update --init --recursive && \
    ./bin/download_external_libraries.py && \
    ./bin/build_external_libraries.py && \
    rm -rf ${MY_MODULE} && \
    mkdir ${MY_MODULE} && \
    ${SRC_DCTRL} ${BLD} all

# THE END
ENV DEBIAN_FRONTEND teletype
