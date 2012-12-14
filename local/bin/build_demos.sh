#!/bin/bash

# always assume that we are in local/bin/
BASEDIR=$(cd $(dirname ${0}) && cd ../.. && pwd)
cd $BASEDIR \
  && test -f PATH \
  && source PATH

echo -n "building demos... "
mkdir -p $BASEDIR/demos \
  ; cd $BASEDIR/demos \
  && cmake ../dune-detailed-discretizations &> /dev/null \
  && make examples_elliptic_continuousgalerkin &> /dev/null \
  && echo "done!" \
  && echo "go to $BASEDIR/demos and run one of the following:" \
  && echo "  cd examples/elliptic/ && ./examples_elliptic_continuousgalerkin" \
  || echo "failed!"

exit 0
