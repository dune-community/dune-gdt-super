#!/usr/bin/env python3

import os
import subprocess
import sys


cc_mapping = {'gcc-5': 'g++-5', 'gcc-6': 'g++-6', 'clang-3.8': 'clang++-3.8', 'clang-3.9': 'clang++-3.9'}

def update(branch, cc):
    gdt_super_dir = os.path.join(os.path.dirname(__file__), '..', '..',)
    dockerfile = os.path.join(os.path.dirname(__file__), 'dune-gdt-testing.docker')

    os.chdir(gdt_super_dir)

    cxx = cc_mapping[cc]
    subprocess.check_call(['docker', 'build', '-f', '.travis/docker/dune-gdt-testing.docker',
                        '-t', 'dunecommunity/dune-gdt-testing:{}_{}'.format(cc, branch), '--build-arg', 'cc={}'.format(cc),
                        '--build-arg', 'cxx={}'.format(cxx), '.'])

if __name__ == '__main__':
    if len(sys.argv) > 2:
        ccs = [sys.argv[1]]
        branches = [sys.argv[2]]
    else:
        ccs = list(cc_mapping.keys())
        branches = ['master']

    for b in branches:
        for c in ccs:
            update(b, c)

    subprocess.check_call(['docker', '--log-level="debug"', 'images'])
    subprocess.check_call(['docker', '--log-level="debug"', 'push', 'dunecommunity/dune-gdt-testing'])
