#!/bin/bash

# create log dir
LOG_DIR=$(mktemp -d /tmp/dune_demos.XXXXXXXX)
LOG_FILE=$LOG_DIR/test.log
touch $LOG_FILE &> /dev/null
if [ $? != 0 ] ; then
  echo "Error: could not create: $LOG_DIR/test.log." >&2
  exit 1
else
  rm $LOG_FILE &> /dev/null
fi

if [ "X$CC" == "X" ] ; then
  export CC=gcc-4.6
  echo "CC not set, defaulting to CC=$CC"
  echo "(set CC to one of the config.opts.??? postfixes)"
fi

echo -ne "initializing submodules (this may take a while)... "
LOG_FILE=$LOG_DIR/git_submodule.log
git submodule update --init &> $LOG_FILE
if [ $? == 0 ] ; then
  source PATH &>> $LOG_FILE
  echo "done"
else
  echo "failed (see $LOG_FILE for details)" >&2
  exit 1
fi

CC=$CC ./local/bin/startup.sh
