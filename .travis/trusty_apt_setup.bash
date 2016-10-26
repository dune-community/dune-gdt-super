#!/bin/bash

SUDO="sudo -E"
sudo -E apt-get install -y -q python-pip python3-pip libscotchparmetis-dev libmetis-dev
sudo -E pip3 install -U requests virtualenv cpp-coveralls
sudo -E pip install -U requests virtualenv cpp-coveralls
${SUDO} add-apt-repository -y ppa:renemilk/llvm
# minimally necessary updates
${SUDO} apt-get install cpp-4.8 g++-4.8 gcc-4.8 gcc-4.8-base lcov libasan0 libgcc-4.8-dev libstdc++-{5,4.9,4.8}-dev gcc-4.9
${SUDO} apt-get update -qq && ${SUDO} aptitude install -y clang-3.8 clang-format-3.8 clang-3.9 lcov
for i in clang-3.7 clang++-3.7 clang-3.8 clang++-3.8 clang-3.9 clang++-3.9 g++-5 g++-6 gcc-5 gcc-6; do
    ${SUDO} ln -s /usr/bin/ccache /usr/lib/ccache/${i}
done
