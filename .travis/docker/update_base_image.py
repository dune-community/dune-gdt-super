#!/usr/bin/env python3

import os
import subprocess
import sys


def update(branch):
    gdt_super_dir = os.path.join(os.path.dirname(__file__), '..', '..',)
    os.chdir(gdt_super_dir)

    subprocess.check_call(['docker', 'build', '-f', '.travis/docker/testing-base.docker',
                        '-t', 'dunecommunity/testing-base:{}'.format(branch), '.'])
    subprocess.check_call(['docker', 'push', 'dunecommunity/testing-base'])

if __name__ == '__main__':
    if len(sys.argv) > 1:
        branch = sys.argv[1]
    else:
        branch = 'master'

    update(branch)

