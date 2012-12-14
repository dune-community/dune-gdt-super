#!/bin/bash

# always assume that we are in local/bin/
BASEDIR=$(cd $(dirname ${0}) && cd ../.. && pwd)
cd $BASEDIR \
  && test -f PATH \
  && source PATH

CONFIG_OPTS_FILE=config.opts.$CC

cd $BASEDIR && \
  nice ./dune-common/bin/dunecontrol --opts=$CONFIG_OPTS_FILE --only=dune-common all \
  && nice ./dune-common/bin/dunecontrol --opts=$CONFIG_OPTS_FILE --only=dune-geometry all \
  && nice ./dune-common/bin/dunecontrol --opts=$CONFIG_OPTS_FILE --only=dune-grid all \
  && nice ./dune-common/bin/dunecontrol --opts=$CONFIG_OPTS_FILE --only=dune-localfunctions all \
  && nice ./dune-common/bin/dunecontrol --opts=$CONFIG_OPTS_FILE --only=dune-fem all \
  && nice ./dune-common/bin/dunecontrol --opts=$CONFIG_OPTS_FILE --only=dune-stuff all \
  && nice ./dune-common/bin/dunecontrol --opts=$CONFIG_OPTS_FILE --only=dune-grid-multiscale all \
  && nice ./dune-common/bin/dunecontrol --opts=$CONFIG_OPTS_FILE --only=dune-detailed-discretizations all \
  || exit 1

exit 0
