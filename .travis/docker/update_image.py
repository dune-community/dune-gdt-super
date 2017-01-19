#!/usr/bin/env python3

import os
import subprocess
import sys


cc_mapping = {'gcc-5': 'g++-5', 'gcc-6': 'g++-6', 'clang-3.8': 'clang++-3.8', 'clang-3.9': 'clang++-3.9'}
thisdir = os.path.dirname(os.path.abspath(__file__))

def update(branch, cc):
    gdt_super_dir = os.path.join(thisdir, '..', '..',)
    dockerfile = os.path.join(thisdir, 'dune-gdt-testing', 'Dockerfile')

    os.chdir(gdt_super_dir)

    cxx = cc_mapping[cc]
    branch = branch.replace('/', '_')
    subprocess.check_call(['docker', 'build', '-f', dockerfile,
                        '-t', 'dunecommunity/dune-gdt-testing:{}_{}'.format(cc, branch), '--build-arg', 'cc={}'.format(cc),
                        '--build-arg', 'cxx={}'.format(cxx), '.'])

if __name__ == '__main__':
    if len(sys.argv) > 2:
        ccs = [sys.argv[1]]
        branches = [sys.argv[2]]
    else:
        ccs = list(cc_mapping.keys())
        branches = ['master']

    subprocess.check_call(['docker', 'pull', 'dunecommunity/testing-base:master'])
    for b in branches:
        for c in ccs:
            update(b, c)

    subprocess.check_call(['docker', '--log-level="debug"', 'images'])
    subprocess.check_call(['docker', '--log-level="debug"', 'push', 'dunecommunity/dune-gdt-testing'])
