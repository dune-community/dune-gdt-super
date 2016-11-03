#!/usr/bin/env python3

import os
import subprocess


gdt_super_dir = os.path.join(os.path.dirname(__file__), '..', ,'..',)
dockerfile = os.path.join(os.path.dirname(__file__), 'dune-gdt.docker')

os.chdir(gdt_super_dir)
subprocess.check_call(['docker', 'build', '-f', '.travis/docker/dune-gdt.docker',
                       '-t', 'renemilk/dune-testing:gcc5', '.'])
