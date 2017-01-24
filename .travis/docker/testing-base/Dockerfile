FROM ubuntu:16.10
MAINTAINER rene.milk@wwu.de

ENV HOME=/root \
    DEBIAN_FRONTEND=noninteractive \
    SUPERDIR=${HOME}/src \
    PATH=${SUPERDIR}/local/bin:$PATH \
    LD_LIBRARY_PATH=${SUPERDIR}/local/lib64:${SUPERDIR}/local/lib:$LD_LIBRARY_PATH \
    PKG_CONFIG_PATH=${SUPERDIR}/local/lib64/pkgconfig:${SUPERDIR}/local/lib/pkgconfig:${SUPERDIR}/local/share/pkgconfig:$PKG_CONFIG_PATH \
    DUNE_BUILD_DIR=${HOME}/dune_build/ \
    INSTALL_DIR=${HOME}/dune \
    CTEST_OUTPUT_ON_FAILURE=1 \
    DCTL_ARGS="--builddir=${DUNE_BUILD_DIR} " \
    DBG="${DCTL_ARGS} --opts=config.opts/travis.ninja" \
    BLD=${DBG} \
    CLANG_FORMAT='/usr/bin/clang-format-3.8' \
    OPT="-O0"

# mirror selection apparently fails for yakkety :(
#RUN echo "deb mirror://mirrors.ubuntu.com/mirrors.txt yakkety main restricted universe multiverse" > /etc/apt/sources.list && \
#    echo "deb mirror://mirrors.ubuntu.com/mirrors.txt yakkety-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
#    echo "deb mirror://mirrors.ubuntu.com/mirrors.txt yakkety-security main restricted universe multiverse" >> /etc/apt/sources.list && \
#    echo "deb-src mirror://mirrors.ubuntu.com/mirrors.txt yakkety main restricted universe multiverse" >> /etc/apt/sources.list && \
#    echo "deb-src mirror://mirrors.ubuntu.com/mirrors.txt yakkety-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
#    echo "deb-src mirror://mirrors.ubuntu.com/mirrors.txt yakkety-security main restricted universe multiverse" >> /etc/apt/sources.list
#add-apt-repository -y ppa:bzindovic/suitesparse-bugfix-1319687
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential git-core aptitude gfortran cmake cmake-data doxygen \
            python-virtualenv python3-pytest python-pytest libboost-all-dev python-pip libtbb-dev ccache \
            libsuitesparse-dev curl ninja-build python3-requests libeigen3-dev openssh-client \
            libstdc++-6-dev libstdc++-5-dev python-setuptools python3-setuptools libsuperlu-dev \
            clang-3.8 clang-format-3.8 lcov git gcc-5 g++-5 gcc-6 g++-6 wget unzip vim bash-completion \
            libmetis-dev python3-pip libfftw3-mpi-dev libfftw3-bin libfftw3-dev libopenmpi-dev openmpi-bin \
            python-pip python3-pip libmetis-dev clang-3.9 clang-format-3.9 texlive-latex-base doxygen-latex \
            cmake-curses-gui gdb rsync libscotchparmetis-dev python3-virtualenv && \
    pip3 install -U requests virtualenv cpp-coveralls && \
    pip install -U requests virtualenv cpp-coveralls && \
    mkdir -p ~/.ssh/ && ssh-keyscan -H github.com >> ~/.ssh/known_hosts && \
    git config --global push.default simple
