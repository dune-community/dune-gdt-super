#!/bin/bash

for i in la functions common grid ; do 
  pushd dune-xt-$i
  git checkout master
  git pull --rebase
  popd
done
