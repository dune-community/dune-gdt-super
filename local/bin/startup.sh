#!/bin/bash

# create log dir
LOG_DIR=$(mktemp -d /tmp/dune_rb_demos.XXXXXXXX)
LOG_FILE=$LOG_DIR/test.log
touch $LOG_FILE &> /dev/null
if [ $? != 0 ] ; then
  echo "Error: could not create: $LOG_DIR/test.log." >&2
  exit 1
else
  rm $LOG_FILE &> /dev/null
fi

echo -ne "writing path definitions... "
LOG_FILE=$LOG_DIR/gen_PATH.log
./local/bin/gen_PATH.sh &> $LOG_FILE
if [ $? == 0 ] ; then
  source PATH &>> $LOG_FILE
  echo "done"
else
  echo "failed (see $LOG_FILE for details)" >&2
  exit 1
fi

echo -ne "initializing submodules (this may take a while)... "
LOG_FILE=$LOG_DIR/initialize_submodules.log
git submodule update --init &> $LOG_FILE
if [ $? == 0 ] ; then
  echo "done"
else
  echo "failed (see $LOG_FILE for details)" >&2
  exit 1
fi

echo -ne "downloading dune modules (this may take a while)... "
LOG_FILE=$LOG_DIR/download_dune_modules.log
./local/bin/download_dune_modules.sh &> $LOG_FILE
if [ $? == 0 ] ; then
  echo "done"
else
  echo "failed (see $LOG_FILE for details)" >&2
  exit 1
fi

echo -ne "downloading external libraries (this may take a while)... "
LOG_FILE=$LOG_DIR/download_external_libraries.log
./local/bin/download_external_libraries.sh &> $LOG_FILE
if [ $? == 0 ] ; then
  echo "done"
else
  echo "failed (see $LOG_FILE for details)" >&2
  exit 1
fi

echo -ne "building external libraries (this may take a while)... "
LOG_FILE=$LOG_DIR/build_external_libraries.log
./local/bin/build_external_libraries.py &> $LOG_FILE
if [ $? == 0 ] ; then
  echo "done"
else
  echo "failed (see $LOG_FILE for details)" >&2
  exit 1
fi

echo -ne "building dune modules (this may take a while)... "
LOG_FILE=$LOG_DIR/build_dune_modules.log
./local/bin/build_dune_modules.sh &> $LOG_FILE
if [ $? == 0 ] ; then
  echo "done"
else
  echo "failed (see $LOG_FILE for details)" >&2
  exit 1
fi

./local/bin/build_demos.sh
