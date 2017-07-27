#!/usr/bin/env python3

import os
import subprocess
import sys

import logging

cc_mapping = {'gcc': 'g++', 'clang': 'clang++'}
thisdir = os.path.dirname(os.path.abspath(__file__))


def update(tests, cc):
    gdt_super_dir = os.path.join(thisdir, '..', '..',)
    dockerfile = os.path.join(thisdir, 'run_tests', 'Dockerfile')
    with open(dockerfile, 'wt') as out:
        out.write(open(dockerfile+'.in').read().replace('COMPILER', cc))

    os.chdir(gdt_super_dir)

    repo = 'dunecommunitylocal/local-gdt-testing_{}_{}'.format(cc, tests)
    subprocess.check_call(['docker', 'build', '-f', dockerfile,
                        '-t', repo, '--build-arg', 'tests={}'.format(tests),
                        os.path.join(thisdir, 'run_tests')])


def update_base(cc):
    gdt_super_dir = os.path.join(thisdir, '..', '..',)
    dockerfile = os.path.join(thisdir, 'run_tests_base', 'Dockerfile')

    os.chdir(gdt_super_dir)

    cxx = cc_mapping[cc]
    try:
        subprocess.check_call(['docker', 'build', '-f', dockerfile,
                            '-t', 'dunecommunitylocal/gdt-testing_base_{}'.format(cc), '--build-arg',
                           'cc={}'.format(cc),
                            '--build-arg', 'cxx={}'.format(cxx),
                            gdt_super_dir])
    except subprocess.CalledProcessError as e:
        logging.error('failed: {} image build {}'.format(cc, tests))
        return True
    return False

def run(tests, cc):
    repo = 'dunecommunitylocal/local-gdt-testing_{}_{}'.format(cc, tests)
    cmd = ['docker', 'run', '-t', repo]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        logging.error('failed: {} tests {}'.format(cc, tests))
        logging.error('rerun "{}" to examine'.format(' '.join(cmd)))


if __name__ == '__main__':
    if len(sys.argv) > 2:
        ccs = [sys.argv[1]]
        tests = [sys.argv[2]]
    else:
        ccs = list(cc_mapping.keys())
        tests = range(13)

    subprocess.check_call(['docker', 'pull', 'dunecommunity/testing-base:latest'])
    stop = False
    for c in ccs:
        update_base(c)
        for b in tests:
            stop = stop or update(b, c)
    if stop:
        logging.error('not running tests because at least on image build failed')
        sys.exit(1)
    for c in ccs:
        for b in tests:
            run(b, c)