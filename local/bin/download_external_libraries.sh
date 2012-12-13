#!/bin/bash

# always assume that we are in local/bin/
BASEDIR=$(cd $(dirname ${0}) && cd ../.. && pwd)
cd $BASEDIR \
  && test -f PATH \
  && source PATH

SRCDIR=$BASEDIR/local/src

mkdir -p $SRCDIR
if [ $? != 0 ] ; then
  echo "Error, could not create directory: $SRCDIR"  >&2
  exit 1
fi

function download {
  SRC="${1}"
  DEST="${2}"
    echo -n "downloading '$DEST':"
  if [ -e "${DEST}" ] ; then
    echo " done (does already exist)"
  else
    echo ""
    wget -O $DEST $SRC
    if [ $? != 0 ] ; then
      echo echo "Error, could not download: '$DEST'"
      exit 1
    fi
  fi
} # function download

# alugrid
cd $SRCDIR \
  && download http://aam.mathematik.uni-freiburg.de/IAM/Research/alugrid/ALUGrid-1.50.tar.gz ALUGrid-1.50.tar.gz

# eigen
cd $SRCDIR \
  && download http://bitbucket.org/eigen/eigen/get/3.1.0.tar.gz eigen-3.1.0.tar.gz

## boost
#cd $SRCDIR \
  #&& download http://wwwmath.uni-muenster.de/u/felix.albrecht/mirror/boost_1_50_0.tar.gz boost_1_50_0.tar.gz

exit 0
