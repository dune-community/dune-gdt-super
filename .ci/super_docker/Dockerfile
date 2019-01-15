ARG BASE
FROM dunecommunity/testing-base_${BASE}:42c68f6

ENV CC=gcc \
    CXX=g++

USER root
ADD . ${SUPERDIR}/
WORKDIR ${SUPERDIR}
RUN chown dune-ci -R ${SUPERDIR}
USER dune-ci

# run our local scripts look for an OPTS env entry
RUN ./bin/download_external_libraries.py 1> /dev/null && \
    ./bin/build_external_libraries.py 1> /dev/null && \
    . environments/${DXT_ENVIRONMENT}/PATH.sh && \
    ${SRC_DCTRL} ${BLD} all
