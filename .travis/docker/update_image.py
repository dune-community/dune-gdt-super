#!/usr/bin/env python3

import os
import subprocess
import sys


branch = sys.argv[2]
cc = sys.argv[1]

gdt_super_dir = os.path.join(os.path.dirname(__file__), '..', ,'..',)
dockerfile = os.path.join(os.path.dirname(__file__), 'dune-gdt.docker')

os.chdir(gdt_super_dir)
cc_mapping = {'gcc-5': 'g++-5', 'gcc-6': 'g++-6', 'clang-3.8': 'clang++-3.8', 'clang-3.9': 'clang++-3.9'])

cxx = cc_mapping[cx]
subprocess.check_call(['docker', 'build', '-f', '.travis/docker/dune-gdt.docker',
                       '-t', 'renemilk/dune-testing:{}_{}'.format(cc), '--build-arg cc={}'.format(cc), '--build-arg cxx={}'.format(cxx), '.'])
