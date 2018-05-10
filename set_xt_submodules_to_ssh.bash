#!/bin/bash

git submodule foreach 'git remote set-url origin $(git config --get remote.origin.url | sed "s;https://github.com/dune-community/;git@github.com:dune-community/;g")'
