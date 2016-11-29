FROM ubuntu:16.10
MAINTAINER rene.milk@wwu.de

ENV HOME /root
ENV DEBIAN_FRONTEND noninteractive

ENV SUPERDIR ${HOME}/src
ENV PATH ${SUPERDIR}/local/bin:$PATH
ENV LD_LIBRARY_PATH ${SUPERDIR}/local/lib64:${SUPERDIR}/local/lib:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH ${SUPERDIR}/local/lib64/pkgconfig:${SUPERDIR}/local/lib/pkgconfig:${SUPERDIR}/local/share/pkgconfig:$PKG_CONFIG_PATH
ENV DUNE_BUILD_DIR ${HOME}/dune_build/
ENV INSTALL_DIR ${HOME}/dune
ENV CTEST_OUTPUT_ON_FAILURE 1
ENV DCTL_ARGS "--builddir=${DUNE_BUILD_DIR} --use-cmake"
ENV DBG "${DCTL_ARGS} --opts=config.opts/travis.ninja"
ENV BLD ${DBG}
ENV CLANG_FORMAT '/usr/bin/clang-format-3.8'
ENV OPT "-O0"

# mirror selection apparently fails for yakkety :(
#RUN echo "deb mirror://mirrors.ubuntu.com/mirrors.txt yakkety main restricted universe multiverse" > /etc/apt/sources.list && \
#    echo "deb mirror://mirrors.ubuntu.com/mirrors.txt yakkety-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
#    echo "deb mirror://mirrors.ubuntu.com/mirrors.txt yakkety-security main restricted universe multiverse" >> /etc/apt/sources.list && \
#    echo "deb-src mirror://mirrors.ubuntu.com/mirrors.txt yakkety main restricted universe multiverse" >> /etc/apt/sources.list && \
#    echo "deb-src mirror://mirrors.ubuntu.com/mirrors.txt yakkety-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
#    echo "deb-src mirror://mirrors.ubuntu.com/mirrors.txt yakkety-security main restricted universe multiverse" >> /etc/apt/sources.list
RUN apt-get update
RUN cat /etc/apt/sources.list
#for add-apt-repo
RUN apt-get install -y software-properties-common
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential git-core aptitude gfortran cmake cmake-data doxygen \
    texlive-base python-virtualenv python3-pytest python-pytest libboost-all-dev python-pip libtbb-dev ccache \
    libsuitesparse-dev curl ninja-build python3-requests libeigen3-dev \
    libstdc++-6-dev libstdc++-5-dev python-setuptools python3-setuptools libsuperlu-dev \
    clang-3.8 clang-format-3.8 lcov git gcc-6 g++-6 wget unzip vim bash-completion \
    libmetis-dev python3-pip libfftw3-mpi-dev libfftw3-bin libfftw3-dev libopenmpi-dev openmpi-bin \
    python-pip python3-pip libmetis-dev clang-3.9 clang-format-3.9 openssh-client \
    cmake-curses-gui gdb rsync
RUN pip3 install -U requests virtualenv cpp-coveralls
RUN pip install -U requests virtualenv cpp-coveralls
# this skips ssh trust question on first connect
RUN mkdir -p ~/.ssh/ && ssh-keyscan -H github.com >> ~/.ssh/known_hosts
RUN git config --global push.default simple
