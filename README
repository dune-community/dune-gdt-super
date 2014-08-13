# This file is part of the dune-gdt-demos project:
#   http://users.dune-project.org/projects/dune-gdt-demos
# Copyright holders: Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

dune-gdt-demos is a git supermodule which serves as a demonstration module
for dune-gdt (http://users.dune-project.org/projects/dune-gdt). This module
provides the correct versions of all relevant DUNE
(http://www.dune-project.org) modules and external libraries as git
submodules. It also provides a git submodule located in local/bin which
containes some helper scripts to download and build the demos.

To get started, clone this repository and initialize the submodules, i.e.:

    git clone http://users.dune-project.org/repositories/projects/dune-gdt-demos.git
    cd dune-gdt-demos
    git submodule update --init

You can now check if one of the config.opts.?? files is named after a compiler
that is available on your system. If not, copy an existing file and edit it
accordingly (mainly CC, CXX and F77), i.e.:

    cp config.opts.gcc config.opts.mycppcompiler

Please note that DUNE in general is known to work best with gcc. dune-gdt
is tested to work with gcc-4.6 and clang-3.1-8. You can now set the CC
environment variable to match your compiler and generate a PATH.sh file, i.e.:

    CC=gcc ./local/bin/gen_path.py
    source PATH.sh

It is convenient to source this PATH.sh file from now on whenever you want to
work on dune-gdt, since many of the provided scripts require the CC variable
to be set correctly.

If your system has all dependencies installed you can now build everything by
calling:

    ./local/bin/startup.sh

This process can take several minutes (up to hours, depending on your system)
and should present you with details on how to run the demos, once it is
finished.
