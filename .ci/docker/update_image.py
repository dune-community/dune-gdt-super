#!/usr/bin/env python3

import os
import subprocess
import sys
import six
import logging
import re
try:
    import docker
except ImportError:
    print('missing module: pip install docker')
    sys.exit(1)
from docker.utils.json_stream import json_stream

cc_mapping = {'gcc': 'g++', 'clang': 'clang++'}
thisdir = os.path.dirname(os.path.abspath(__file__))


def _build(client, **kwargs):
    resp = client.api.build(**kwargs)
    if isinstance(resp, six.string_types):
        return client.images.get(resp)
    last_event = None
    image_id = None
    output = []
    for chunk in json_stream(resp):
        if 'error' in chunk:
            msg = chunk['error'] + '\n' + ''.join(output)
            raise docker.errors.BuildError(msg, chunk)
        if 'stream' in chunk:
            output.append(chunk['stream'])
            match = re.search(
                r'(^Successfully built |sha256:)([0-9a-f]+)$',
                chunk['stream']
            )
            if match:
                image_id = match.group(2)
        last_event = chunk
    if image_id:
        return client.images.get(image_id)
    raise docker.errors.BuildError(last_event or 'Unknown', '')


def update(commit, refname, cc):
    gdt_super_dir = os.path.join(thisdir, '..', '..',)
    dockerfile = os.path.join(thisdir, 'dune-gdt-testing', 'Dockerfile')
    client = docker.from_env(version='auto')

    os.chdir(gdt_super_dir)

    cxx = cc_mapping[cc]
    repo = 'dunecommunity/dune-gdt-testing_base_{}'.format(cc)
    buildargs = {'cc': cc, 'cxx': cxx, 'commit': commit}
    tag = '{}:{}'.format(repo, refname)
    img = _build(client, rm=True, fileobj=open(dockerfile, 'rb'),
                        tag=tag, buildargs=buildargs, nocache=False)
    img.tag(repo, refname)
    img.tag(repo, commit)
    client.images.push(repo, tag=refname)
    client.images.push(repo, tag=commit)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        ccs = [sys.argv[1]]
    else:
        ccs = list(cc_mapping.keys())

    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], universal_newlines=True).strip()
    commit = os.environ.get('DRONE_COMMIT_SHA', head)
    refname = os.environ.get('DRONE_COMMIT_BRANCH', 'master').replace('/', '_')

    logger = logging.getLogger('docker-update')
    logger.error('updating images for {}({}) - {}'.format(refname, commit, ', '.join(ccs)))
    subprocess.check_call(['docker', 'pull', 'dunecommunity/testing-base_debian:latest'])
    for c in ccs:
        try:
            logger = logging.getLogger('docker-update')
            logger.error('updating images for {}'.format(c))
            update(commit, refname, c)
        except docker.errors.BuildError as be:
            print(be.msg)
            break
