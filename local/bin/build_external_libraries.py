#!/usr/bin/env python

#import logging
#logging.debug("blah")
from os.path import (join, exists)
import os
import shutil
import subprocess
import sys

basedir = os.path.abspath(join(os.path.dirname(sys.argv[0]), "..", ".."))
srcdir = join(basedir, "local", "src")
sys.path.append(basedir)
if exists(basedir + "path.py"):
    import path


def extract(tarfile, untared_name, target_name):
    if not exists(join(srcdir, target_name)):
        if not exists(join(srcdir, untared_name)):
            if exists(join(srcdir, tarfile)):
                subprocess.call(["tar", "xzf", tarfile], cwd=srcdir)
            else:
                raise Exception('Error: %s does not exist'
                                % join(srcdir, tarfile))
        shutil.move(join(srcdir, untared_name), join(srcdir, target_name))


# alugrid
extract('ALUGrid-1.50.tar.gz', "ALUGrid-1.50", "alugrid")
alugrid_configure = ['./configure', '--prefix=' + join(basedir, "local")]
try:
    alugrid_configure.append('CC=' + path.MY_CC)
except:
    pass
try:
    alugrid_configure.append('CXX=' + path.MY_CXX)
except:
    pass
try:
    alugrid_configure.append('F77=' + path.MY_F77)
except:
    pass
try:
    alugrid_configure.append('CXXFLAGS="' + path.MY_CXXFLAGS + '"')
except:
    pass
subprocess.call(alugrid_configure,
                cwd=join(srcdir, "alugrid"),
                stdout=sys.stdout, stderr=sys.stderr)
subprocess.call(['make'],
                cwd=join(srcdir, "alugrid"),
                stdout=sys.stdout, stderr=sys.stderr)
subprocess.call(['make', 'install'],
                cwd=join(srcdir, "alugrid"),
                stdout=sys.stdout, stderr=sys.stderr)
# eigen
extract('eigen-3.1.0.tar.gz', 'eigen-eigen-ca142d0540d3', 'eigen')
try:
    os.mkdir(join(basedir, "local", "src", 'eigen', 'build'))
except OSError, os_error:
    if os_error.errno != 17:
        raise os_error
eigen_cmake = ['cmake', '..',
               '-DCMAKE_INSTALL_PREFIX=' + join(basedir, 'local')]
try:
    eigen_cmake.append('-DCMAKE_CXX_COMPILER=' + path.MY_CXX)
except:
    pass
subprocess.call(eigen_cmake,
                cwd=join(srcdir, 'eigen', 'build'),
                stdout=sys.stdout, stderr=sys.stderr)
subprocess.call(['make'],
                cwd=join(srcdir, 'eigen', 'build'),
                stdout=sys.stdout, stderr=sys.stderr)
subprocess.call(['make', 'install'],
                cwd=join(srcdir, 'eigen', 'build'),
                stdout=sys.stdout, stderr=sys.stderr)
