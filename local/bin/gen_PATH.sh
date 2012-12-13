#!/bin/bash

BASEDIR=$(cd $(dirname ${0}) && cd ../.. && pwd)
echo "export BASEDIR=$BASEDIR" > $BASEDIR/PATH
echo "export PATH=$BASEDIR/local/bin:$PATH" >> $BASEDIR/PATH
echo "export LD_LIBRARY_PATH=$BASEDIR/local/lib:$LD_LIBRARY_PATH" >> $BASEDIR/PATH
echo "export PKG_CONFIG_PATH=$BASEDIR/local/lib/pkgconfig:$PKG_CONFIG_PATH" >> $BASEDIR/PATH
echo "export MY_CC=gcc-4.6" >> $BASEDIR/PATH
echo "export MY_CXX=g++-4.6" >> $BASEDIR/PATH
echo "export MY_F77=gfortran-4.6" >> $BASEDIR/PATH
echo "export MY_CXXFLAGS='-std=c++0x -O0 -g -ggdb -DDEBUG'" >> $BASEDIR/PATH
echo "export CC=\$MY_CC" >> $BASEDIR/PATH
echo "export CXX=\$MY_CXX" >> $BASEDIR/PATH
echo "export F77=\$MY_F77" >> $BASEDIR/PATH

echo "PATH = \"$BASEDIR/local/bin:$PATH\"" > $BASEDIR/path.py
echo "LD_LIBRARY_PATH = \"$BASEDIR/local/lib:$LD_LIBRARY_PATH\"" >> $BASEDIR/path.py
echo "PKG_CONFIG_PATH = \"$BASEDIR/local/lib/pkgconfig:$PKG_CONFIG_PATH\"" >> $BASEDIR/path.py
echo "MY_CC = \"gcc-4.6\"" >> $BASEDIR/path.py
echo "MY_CXX = \"g++-4.6\"" >> $BASEDIR/path.py
echo "MY_F77 = \"gfortran-4.6\"" >> $BASEDIR/path.py
echo "MY_CXXFLAGS = \"-std=c++0x -O0 -g -ggdb -DDEBUG\"" >> $BASEDIR/path.py
