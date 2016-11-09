FROM dunecommunity/testing-base:master

# COMPILER Selection (ARGS cannot be uppercase)
ARG cc
ARG cxx
ENV CC $cc
ENV CXX $cxx

ENV MY_MODULE dune-gdt

WORKDIR ${SUPERDIR}
ADD . ${SUPERDIR}

RUN apt-get install -y --no-install-recommends libscotchparmetis-dev

# TRAVIS before_install:
ENV OPTS config.opts/travis.ninja


RUN cd ${SUPERDIR}
RUN git submodule foreach git reset --hard
RUN git submodule update --init --recursive
RUN git submodule status

ENV PATH /usr/lib/ccache:${PATH}
# our local scripts look for an OPTS env entry
RUN ./local/bin/download_external_libraries.py
RUN ./local/bin/build_external_libraries.py
# ensures ${MY_MODULE} from travis own checkout is used via clean mount point
RUN rm -rf ${MY_MODULE}
RUN mkdir ${MY_MODULE}

# TRAVIS install
ENV SRC_DCTRL ${SUPERDIR}/dune-common/bin/dunecontrol
RUN ${SRC_DCTRL} ${BLD} all



# THE END
ENV DEBIAN_FRONTEND teletype
